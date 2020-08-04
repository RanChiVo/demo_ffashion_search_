
# FFASHION SEARCH DEMO

## Introduction
A fashion image query system is similar to google image. The system is built on an open source visual fashion analysis based on PyTorch - **mmfashion**.
This project is implemented in the form of a website and is the result of the learning  and researching process in this [[article]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf).
- Unlike traditional image retrieval, this feature removes the need to type in keywords and terms into the search box. Instead, users search by submitting an image as their query. Results may include tops of similar images.

## Features
- The image's input will be any fashion image from your device or the address of the image on the website. The search result will be a top of the image similar to the input image.
- The system allows you to enable the use of two models: VGG16 and ResNet50 to check the accuracy of the query results.

## Environment settings

- [Python 3.5+](https://www.python.org/)
- [PyTorch 1.0.0+](https://pytorch.org/)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [Flask==1.1.2](https://flask.palletsprojects.com/en/1.1.x/installation/)
- [Bootstrap 4](https://github.com/greyli/bootstrap-flask)
## Prepair dataset and model

- Using two datasets: [DeepFashion - Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) and [DeepFashion - In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). The preparation of data just follow the instructions of mmfashion in this section [[data]](https://github.com/open-mmlab/mmfashion/blob/master/docs/DATA_PREPARATION.md).
- Using available models of mmfashion. You can download and follow the intructions of mmffashion [[model]](https://github.com/open-mmlab/mmfashion/blob/master/docs/DATA_PREPARATION.md)

## Installation
You need to work with a virtual environment such as anaconda and then install flask.
```sh
conda install -c anaconda flask
```
```sh
conda install -c conda-forge flask-bootstrap
```

```sh
git clone --recursive https://github.com/open-mmlab/mmfashion.git
cd mmfashion
python setup.py install
```
```sh
cd flaskr
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
```
## Instructor
- Trần Anh Dũng

## Student
- Võ Thị Một - 16520756
- Class: SE121.K21.PMCL
