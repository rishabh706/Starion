import albumentations as A 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# def check_bbox(bbox):
#     bbox=list(bbox)
#     for boxes in bbox:
#         for i in range(4):
#             if (boxes[i]<0):
#                 boxes[i]=0
#             elif (boxes[i]>1):
#                 boxes[i]=1
        
#         final_bbox=tuple(boxes)

#         return final_bbox

annotations=[os.path.join("annotated_images",i) for i in os.listdir("annotated_images") if i.endswith(".txt")]
    
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(p=0.2)
],bbox_params=A.BboxParams(format='yolo'))
class_label=["ok","ng"]

    
for i in tqdm(range(50)):
        
    for gt in annotations:
        with open(gt,"r") as f:
            coords=[i.strip().split(" ") for i in f.readlines()]
            # coords=np.array(coords).astype("float")
            # bboxes=[]
            # class_labels=[i[0] for i in f.readlines()]
            coordinates=[list(map(float,i[1:]))+[int(i[0])] for i in coords]
                
        try:
            
            img_name=gt.replace(".txt",".bmp")
            img=cv2.imread(img_name)
            transformed = transform(image=img,bboxes=coordinates)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            name=os.path.join("augmented_stacked_images",f"{i}_{os.path.basename(gt)}")
            img_name=name.replace(".txt",".bmp")
            with open(name,"w") as f:
                for cords in transformed_bboxes:
                    f.write(f"{cords[-1]} {cords[0]} {cords[1]} {cords[2]} {cords[3]}\n")
                    # # transformed_labels=transformed["class_labels"]
                    # H,W,C=transformed_image.shape
                    # x,y,w,h=cords[:4]*np.array([W,H,W,H])
                    # x_min,y_min,x_max,y_max=int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)
                    # label=class_label[int(cords[-1])]

            cv2.imwrite(img_name,transformed_image)
            
        except Exception as e:
            pass
    #     cv2.rectangle(transformed_image,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    # plt.imshow(transformed_image)
    # plt.show()
