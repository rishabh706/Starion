import numpy as np
import time
import cv2
import os
import sys
import matplotlib.pyplot as plt
import glob


class WM:
    def __init__(
        self,
        weightsPath="yolov3_files/yolov3_best.weights",
        configPath="yolov3_files/yolov3.cfg",
        labelsPath="yolov3_files/label.names",
        debug=False,
    ):
        self.weightsPath = weightsPath
        self.configPath = configPath
        self.labelsPath = labelsPath
        self.debug = debug

    def load_network(self):
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.net = net

    def predict(self, img):
        try:

            LABELS = open(self.labelsPath).read().strip().split("\n")
            a = 0
            image = img
            (H, W) = image.shape[:2]
            # determine only the *output* layer names that we need from YOLO
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(
                image, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            start = time.time()
            layerOutputs = self.net.forward(ln)
            end = time.time()
            # show timing information on YOLO
            ##      print("[INFO] YOLO took {:.6f} seconds".format(end - start))
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.8:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height), centerX, centerY])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
            ##      print(idxs)
            cords = []

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keepin
                for i in idxs.flatten():
                    detection_dict = {}
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cx, cy = boxes[i][4], boxes[i][5]
                    text = "{}".format(LABELS[classIDs[i]])
                    confidence = round(confidences[i], 2)
                    centerx, centery = x + (w / 2), y + (w / 2)
                    a, b = x, y
                    c, d = x + w, y + h
                    H, W = image.shape[0], image.shape[1]
                    detection_dict["okng"] = text
                    detection_dict["x1"] = str(round(cx / W, 2))
                    detection_dict["y1"] = str(round(cy / H, 2))
                    detection_dict["width"] = str(round(w / W, 2))
                    detection_dict["height"] = str(round(h / H, 2))

                    cords.append(detection_dict)

            return {"_ResponseList": cords}

        except Exception as e:
            print(e)

            return None

    def cropandstack(self, im):
        splits = 4
        first = im[0 : 200 * splits, 200:400]
        second = im[0 : 200 * splits, 1495:1595]

        return np.hstack((first, second))

    def get_predictions(self, imgPath,save=False):
        splits = 4
        label_map={"ok":0,"ng":1}

        final_preds = []
        image_name = os.path.basename(imgPath)
        img = cv2.imread(imgPath)
        x, y, w, h = 0, 0, 1936, 200 * splits
        img_splits = []
        for i in range(int(48 / splits)):
            im = img[y : y + h, x : x + w]

            im = self.cropandstack(im)
            img_splits.append(im)

            y = y + 200 * splits

        for i in range(4):
            img2concat = img_splits[3 * i : 3 * (i + 1)]
            im = np.hstack(img2concat)
            cords = self.predict(im)
            H, W, _ = im.shape

            txt_name=f"detected//{i}_{image_name}".replace(".bmp",".txt")
            img_name=f"detected//{i}_{image_name}".replace(".bmp",".png")

            if save==True:
                with open(txt_name ,"w") as file:
                    for ind,c in enumerate(cords["_ResponseList"]):
                        label = c["okng"]
                            

                        final_preds.append(label)

                        #################### To show the results ##############################
                        x1 = float(c["x1"]) * W
                        y1 = float(c["y1"]) * H
                        w = float(c["width"]) * W
                        h = float(c["height"]) * H

                        xmin, ymin, xmax, ymax = (
                            int(x1) - int(w / 2),
                            int(y1) - int(h / 2),
                            int(x1) + int(w / 2),
                            int(y1) + int(h / 2),
                        )

                        
                        file.write(f"{label_map[label]} {c['x1']} {c['y1']} {c['width']} {c['height']}\n")
    ##                    if label == "ng":
    ##
    ##                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    ##
    ##                    elif label == "ok":
    ##
    ##                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # cv2.imwrite(f"images//{time.time()}.png",im)
                    cv2.imwrite(img_name, im)



            else:

                
                for ind,c in enumerate(cords["_ResponseList"]):
                    label = c["okng"]
                        

                    final_preds.append(label)

                    #################### To show the results ##############################
                    x1 = float(c["x1"]) * W
                    y1 = float(c["y1"]) * H
                    w = float(c["width"]) * W
                    h = float(c["height"]) * H

                    xmin, ymin, xmax, ymax = (
                        int(x1) - int(w / 2),
                        int(y1) - int(h / 2),
                        int(x1) + int(w / 2),
                        int(y1) + int(h / 2),
                    )

                        

        if "ng" in final_preds:

            return {"okng": "ng"}

        elif "ok" in final_preds and len(final_preds) > 0:

            return {"okng": "ok"}

        elif "ok" in final_preds:

            return {"okng": "ok"}

        else:
            
            return {"okng": "null"}


if __name__ == "__main__":

    model = WM()
    model.load_network()
    for imgPath in glob.glob("dataset_linescan/*.bmp"):
        print(imgPath)

        print(model.get_predictions(imgPath))
