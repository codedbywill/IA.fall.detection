from ultralytics import YOLO


#treinos padrões dentro do yolo
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

#passando o caminho do treinamento e treinando por 5 epocas, os outros parametros sao para velocidade do treino.
result = model.train(data='./data.yaml', epochs=5, imgsz=608, batch = 4)
