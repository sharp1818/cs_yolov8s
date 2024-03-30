import mss
import numpy as np
import keyboard
import serial
import pyautogui
from PIL import Image
from ultralytics import YOLO
import serial.tools.list_ports

arduino = None
DETECTION_Y_PORCENT = 0.8

def detect_arduino_port():
    arduino_ports = []
    for port in serial.tools.list_ports.comports():
        if 'Arduino' in port.manufacturer:
            arduino_ports.append(port.device)
    return arduino_ports

def init_arduino():
    arduino_ports = detect_arduino_port()
    if arduino_ports:
        port = arduino_ports[0]
        return serial.Serial(port, 115200, timeout=0)
    else:
        print("No se detectaron puertos de Arduino. Verifica la conexión.")
        return(arduino)

def aim(bbox, arduino):
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerX = int((bbox[2] + bbox[0]) / 2)
    centerY = int((bbox[3] + bbox[1]) / 2 - (bbox[3] - bbox[1]) / 2 * DETECTION_Y_PORCENT)
    mouse_x, mouse_y = pyautogui.position()
    moveX = int((centerX - mouse_x))
    moveY = int((-centerY + mouse_y))
    return arduino.write((str(moveX) + ":" + str(moveY) + 'x').encode())

def main():
    arduino = init_arduino()
    if not arduino:
        return

    model = YOLO('.\\train\\best.pt')

    with mss.mss() as sct: 
        monitor_number = 1
        mon = sct.monitors[monitor_number]
        width = 1920
        height = 1080
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": width,
            "height": height,
            "mon": monitor_number,
        }
    
        classes = [0]
        
        while True:
            img = np.array(Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb))
            results = model.track(img, persist=True, conf=0.8, iou=0.7, classes=classes)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy().astype(int)
                box = boxes[confidences.argmax()]
                aim(box, arduino)
                
            if keyboard.is_pressed('j'):
                classes = [1, 2]
                print('ct')
            if keyboard.is_pressed('k'):
                classes = [3, 4]
                print('t')
            if keyboard.is_pressed('o'):
                classes = [0]
                print('none')
            if keyboard.is_pressed('q'):
                break

if __name__ == "__main__":
    main()