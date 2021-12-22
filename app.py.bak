
  import flask as Flask
  import torch 
  import pandas as pd
  
  import numpy as np
  import cv2
  import matplotlib.pyplot as plt
  input ='original.jpg"
  app = Flask(__name__)
  @app.route('/predict', methods=['POST' , 'GET'])
  def predict():
    model = torch.load('/GFPGAN-master/experiments/pretrained_models/GFPGANv1.pth' )
        with torch.no_grad():
            predictions= model(input)
  def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    # read images
  img1 = imread(predictions)
    # show images
  fig = plt.figure(figsize=(25, 0))

  ax1 = fig.add_subplot(1, 2, 1) 
  ax1.imshow(img1)
  ax1.axis('off')
  if __name__ == ('__main__'):
    app.run(debug= True)

