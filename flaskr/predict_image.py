from __future__ import division
import argparse
import os
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.core import ClothesRetriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever
from mmfashion.utils import get_img_tensor
from werkzeug.utils import secure_filename
from torch.utils.cpp_extension import CUDA_HOME
import urllib
import functools
import urllib3
import time
## Importing Necessary Modules

import requests # to get image from the web
import shutil # to save it locally
from flask import send_from_directory
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, send_from_directory, jsonify
)
from werkzeug.security import check_password_hash, generate_password_hash

bp = Blueprint('predict', __name__)

gallery_path = 'static/img/'
path = 'flaskr/static'
current_path = ''
extensions = ["JPEG", "JPG", "PNG", "GIF"]
size = 0.5 * 1024 * 1024
topk = [15]
new_model = None
cfg = None 
check_point = {"vgg":"checkpoints/Inshop/vgg16GlobalPooling/epoch_100.pth",
                "resnet":"checkpoints/Inshop/ResNet-50GlobalPooling/latest.pth"}
config = {"vgg":'configs/retriever_in_shop/global_retriever_vgg.py',
                "resnet":"configs/retriever_in_shop/global_retriever_resnet.py"}
model_types = {"vgg":0, "resnet":1}

current_model = model_types["vgg"]

is_loaded_first = False

def _process_embeds(dataset, model, cfg, use_cuda=True):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    embeds = []
    with torch.no_grad():
        for data in data_loader:
            if use_cuda:
                img = data['img'].cuda()
            embed = model(img, landmark=data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in extensions:
        return True
    else:
        return False

def allowed_image_filesize(filesize):

    if int(filesize) <=size:
        return True
    else:
        return False    

def load_model(model_type):
    global current_model
    current_model = model_types[model_type]
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    global cfg
    cfg = Config.fromfile(config[model_type])
    global new_model
    
    new_model = build_retriever(cfg.model)
    load_checkpoint(new_model, check_point[model_type])
    new_model.cuda()
    new_model.eval()

@bp.route('/')
def upload_form():
    global new_model
    global cfg
    global is_loaded_first
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = Config.fromfile(config["vgg"])
    
    if is_loaded_first==False:
        is_loaded_first = True
        new_model = build_retriever(cfg.model)
        load_checkpoint(new_model, check_point["vgg"])
        new_model.cuda()
        new_model.eval()
        return render_template('index.html')

@bp.route('/upload_online', methods=('GET', 'POST'))
def upload_image_online():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    image_paths_topk = []

    global current_path
    if request.method == "POST":
        print("METHOD: ", request.method)
        print("request:", request.json['url_online'])
        image_url = request.json['url_online']
        img_name = os.path.basename(image_url)
        img_path = os.path.join(path, img_name)
        print("image_path", img_path)
        urllib.request.urlretrieve(image_url, img_path)

 
        model = request.json['model']

        print('request:',img_path)

        print('model:',model)
        
        if current_model != model_types[model]:
            load_model(model)

        img_tensor = get_img_tensor(img_path, use_cuda=True)

        query_feat = new_model(img_tensor, landmark=None, return_loss=False)
        query_feat = query_feat.data.cpu().numpy()

        gallery_set = build_dataset(cfg.data.gallery)
        gallery_embeds = _process_embeds(gallery_set, new_model, cfg)

        retriever = ClothesRetriever(cfg.data.gallery.img_file, cfg.data_root,
                                        cfg.data.gallery.img_path)
        image_paths_topk = retriever.show_retrieved_images(query_feat, gallery_embeds)
        
        image_paths_topk = list(map(lambda x: gallery_path + x, image_paths_topk))

        image_paths_topk.append(url_for('static', filename=img_name))

        print("image_paths:", image_paths_topk) 

    return jsonify(images=image_paths_topk)
    
    
@bp.route('/upload', methods=('GET', 'POST'))
def upload_image():
    global current_path
    if request.method == "POST":
        print("METHOD: ", request.method)
        print(request.files)
        if request.files:
            print("file ok")
            image = request.files['file_up']

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(path, filename))
                current_path = url_for('static', filename=filename)
                print('current_path',current_path)
                return jsonify(url=current_path)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)
        else:
            print("no file")
    return 'no file'

    
@bp.route('/search', methods=['POST'])
def search():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    if request.method == "POST":
        if request.json:
            request_path =  'flaskr' + request.json['url']

            model = request.json['model']

            print('request:',request_path)

            print('model:',model)
            
            if current_model != model_types[model]:
                load_model(model)

            image_paths_topk = []

            img_tensor = get_img_tensor(request_path, use_cuda=True)

            query_feat = new_model(img_tensor, landmark=None, return_loss=False)
            query_feat = query_feat.data.cpu().numpy()

            gallery_set = build_dataset(cfg.data.gallery)
            gallery_embeds = _process_embeds(gallery_set, new_model, cfg)

            retriever = ClothesRetriever(cfg.data.gallery.img_file, cfg.data_root,
                                            cfg.data.gallery.img_path, topks=topk)
            image_paths_topk = retriever.show_retrieved_images(query_feat, gallery_embeds)
            
            image_paths_topk = list(map(lambda x: gallery_path + x, image_paths_topk))

            print("image_paths:", image_paths_topk)

    return jsonify(images=image_paths_topk)
