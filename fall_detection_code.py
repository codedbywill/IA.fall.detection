from plyer import notification
from ultralytics import YOLO
from twilio.rest import Client
import numpy as np
import os
import cv2

account_sid = 'ACc6ccdbcf4ced2e373e56ff0976402f84'
auth_token = 'd8f2b097328038c41908ca9e7a0c1430'
client = Client(account_sid, auth_token)

#print(message.sid)
# Crie o cliente da Twilio
client = Client(account_sid, auth_token)
# Número de telefone do Twilio para WhatsApp
twilio_whatsapp_number = 'whatsapp:+14155238886'
# Número de telefone de destino (deve ser verificado no sandbox)
to_whatsapp_number = 'whatsapp:+5519993362578'
#mensagem passada
message_body = 'ACIDENTE INDENTIFICADO. \n\n POR FAVOR VERIFIQUE SUAS CAMERAS IMEDIATAMENTE!\n\n LIGUE PARA UMA AMBULÂNCIA SE NECESSÁRIO (192)'

# Criar a notificação
notification_title = "Queda Acidental na Residência"
notification_message = "Prezado(a) Celso,\n\nInformamos que houve uma queda acidental em sua residência. Favor verificar o local e tomar as medidas cabíveis.\n\nEm caso de dúvidas ou urgências, entre em contato conosco pelo número [Número de Telefone]."


# Definir o diretório de vídeos
VIDEOS_DIR = os.path.join('.', 'test')
# Caminhos de entrada
video_path = os.path.join(VIDEOS_DIR, 'velho_caindo2.avi')
#Saída de video
video_path_out = '{}_out.mp4'.format(video_path)

# Verificar se o arquivo de vídeo existe
if not os.path.isfile(video_path):
    print(f"Erro: O arquivo de vídeo '{video_path}' não existe.")

else:
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    
    # Verificar se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo '{video_path}'.")
    else:
        ret, frame = cap.read()
        
        # Verificar se o frame foi lido com sucesso
        if not ret:
            print(f"Erro: Não foi possível ler o frame do vídeo '{video_path}'.")
        else:
            # Obter as dimensões de cada frame 
            H, W, _ = frame.shape
            
            # Caminho do modelo treinado 
            model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')
            
            # Carregar o modelo treinado YOLO
            model = YOLO(model_path)
            
            #estimativa de queda
            confidence_threshold = 0.40

            #estimativa de queda
            alert_threshold = 0.64
            
            first_fall_frame = None  # Variável para armazenar o índice do primeiro frame com queda detectada 
            first_fall_frame_image = None  # Variável para armazenar a imagem do primeiro frame com queda detectada
            frame_index = 0  # Variável para contar os frames
            cont_detected = 0
            
            # Processar cada frame do vídeo
            while ret:
                frame_index += 1
                results = model(frame)[0]
                
                #pegando as variaveis que a IA PRODUZIU
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = result

                    #SE ESTIMATIVA DE QUEDA > 0.40 ELE VAI IMPRIMIR O RETANGULO. 
                    if confidence > confidence_threshold:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f"{results.names[int(class_id)].upper()} {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cont_detected += 1

                        if cont_detected == 1: #se contagem de quedas for igual a 1
                            first_fall_frame = frame_index
                            first_fall_frame_image = frame.copy()  # Salvar a imagem do primeiro frame com detecção de queda
                            cv2.imshow("Possivel Queda Detectada", first_fall_frame_image)
                            cv2.waitKey(0)  # Esperar que uma tecla seja pressionada para continuar
                            #cv2.destroyWindow("Queda Detectada")

                        if confidence > alert_threshold: #SE O NIVEL DE CONFIANCA (estimativa de queda)) > 0.64  alert_threshold.
                            # Exibir a imagem e a mensagem de aviso
                            alert_frame = frame.copy()
                            cv2.putText(alert_frame, "EMERGENCIA", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow("Alerta", alert_frame)

                            #mandar uma mensagem no whatsapp via twilio api
                            message = client.messages.create(
                                from_=twilio_whatsapp_number,
                                body=message_body,
                                to=to_whatsapp_number)
                            cv2.waitKey(0)  # Esperar que uma tecla seja pressionada para continuar
                            cv2.destroyWindow("Alerta")

                            # Exibir a notificação
                            notification.notify(
                                title=notification_title,
                                message=notification_message,
                                app_name="Queda Acidental",
                                timeout=10,  # Tempo de exibição em segundos
                                toast=False  # Não usar notificação estilo toast
                            )

                cv2.imshow("Processando Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                ret, frame = cap.read()
            
            print(f"Quantidade de frames: '{frame_index}'.")
            print(f"Quantidade de frames detectando quedas: '{cont_detected}'.")
            print(f"O primeiro frame com detecção de queda é o frame: {first_fall_frame}.")
            print(f'SMS enviado com o SID: {message.sid}')

            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processamento concluído.")
