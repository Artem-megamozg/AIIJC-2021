# нормальный код начало

import cv2

import numpy as np
import pybullet as p
import time
import pybullet_data

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_teapot",
                        {},
                        "dataset_final/label/*",
                        "dataset_final/images/*")

# test_register_coco_instances = MetadataCatalog.get("my_dataset_teapot")
### Run a pre-trained detectron2 model ###
# We first download an image from the COCO dataset: #

im = cv2.imread("dataset_final/images/3-5.jpg")
cv2.imshow('Image', im)
cv2.waitKey(1)
cv2.destroyWindow('Image')

# Создание конфига

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("my_dataset_teapot"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("my_dataset_teapot")
cfg.DATASETS.TRAIN = ["my_dataset_teapot"]
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

DURATION = 10000
ALPHA = 300

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print("data path: %s " % pybullet_data.getDataPath())
p.setGravity(0, 0, -5)
planeId = p.loadURDF("plane.urdf")
p.setRealTimeSimulation(1)
kukaStartPos = [0, 0, 0]
kukaStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
kukaId = p.loadURDF("resources/ur10-wsg50-realsense.urdf", kukaStartPos, kukaStartOrientation)
# kukaId = p.loadURDF("robot_movement_interface-master/dependencies/ur_description/urdf/ur10_robot.urdf", cubeStartPos, cubeStartOrientation)
teapotId = p.loadURDF("(Unsaved)_description/urdf/(Unsaved).xacro", [
    1, 0.8, 0.01], p.getQuaternionFromEuler([1, 1, 1]), globalScaling=1.8)

numJoints = p.getNumJoints(kukaId)
for joint in range(numJoints):
  print(p.getJointInfo(kukaId, joint))
targetVel = 10  #rad/s
maxForce = 500  #Newton

# for joint in range(1, 18):
#   p.setJointMotorControl(kukaId, joint, p.VELOCITY_CONTROL, targetVel, maxForce)
# for step in range(300):
#   p.stepSimulation()
#
# targetVel = -10
# for joint in range(1, 18):
#   p.setJointMotorControl(kukaId, joint, p.VELOCITY_CONTROL, targetVel, maxForce)
# for step in range(400):
#   p.stepSimulation()
# targetVel = -30
# for joint in range(1, 18):
#   p.setJointMotorControl(kukaId, joint, p.VELOCITY_CONTROL, targetVel, maxForce)
# for step in range(400):
#   p.stepSimulation()
# targetVel = 10
# for joint in range(1, 18):
#   p.setJointMotorControl(kukaId, joint, p.VELOCITY_CONTROL, targetVel, maxForce)
# for step in range(400):
#   p.stepSimulation()
#
# p.getContactPoints(7)


for i in range(DURATION):
    p.stepSimulation()
    time.sleep(1. / 240.)
    teapotPos, teapotOrn = p.getBasePositionAndOrientation(teapotId)
    kukaPos, kukaOrn = p.getBasePositionAndOrientation(kukaId)
    test = p.getLinkState(kukaId, 7)

    print(test[0], test[1])
    x = test[0][0]
    y = test[0][1]
    z = test[0][2]

    # i = test[1][0]
    # j = test[1][1]
    # k = test[1][2]

# камера начало

    viewMatrix = p.computeViewMatrix(
        # cameraEyePosition=[0, 0, 0],
        cameraEyePosition=[x, y, z + 2],
        # cameraEyePosition=kukaPos,
        # cameraTargetPosition=[0, 0, 0],
        # cameraTargetPosition=[i, j, k],
        cameraTargetPosition=[x, y, z],
        # cameraTargetPosition=kukaPos,
        cameraUpVector=[0, 1, 0])

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

    # print(rgbImg)
    # import matplotlib.pyplot as plt
    # plt.imshow(rgbImg)

    # im2 = cv2.imread("dataset_final/images/detail_ed8a4a6d633b6c704a1a57c0fd626e97.jpg")
    # im = segImg
    # im = depthImg
    im = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('ЧАЙНИК', out.get_image()[:, :, ::-1])
    cv2.waitKey()
    cv2.destroyWindow('ЧАЙНИК')
    #добавить вывод в консоль

    oid, lk, frac, pos, norm = p.rayTest(teapotPos, kukaPos)[0]
    # rt = p.rayTest(cubePos, gemPos)
    # print("rayTest: %s" % rt[0][1])
    print("rayTest: Norm: ")
    print(norm)
    p.applyExternalForce(objectUniqueId=kukaId, linkIndex=-1, forceObj=pos
                         , posObj=kukaPos, flags=p.WORLD_FRAME)


    np_img_arr = np.reshape(rgbImg, (height, width, 4))
    # render = rgbImg
# камера конец

    force = ALPHA * (np.array(teapotPos) - np.array(kukaPos))
    p.applyExternalForce(objectUniqueId=kukaId, linkIndex=-1,
                         forceObj=force, posObj=kukaPos, flags=p.WORLD_FRAME)

    # print('Applied force magnitude = {}'.format(force))
    # print('Applied force vector = {}'.format(np.linalg.norm(force)))

p.disconnect()








# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('ЧАЙНИК', out.get_image()[:, :, ::-1])
cv2.waitKey(1)
cv2.destroyWindow('ЧАЙНИК')



# нормальный код окончание


# from detectron2.utils.logger import setup_logger
#
# setup_logger()
# import some common libraries

# import some common detectron2 utilities

# import numpy as np
# import os, json, cv2, random

# im_m = cv2.imread("dataset_v2/images/*")

# for i in range(5):
#     im = im_m[i]
#     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
#     cv2.waitKey(1)
#     cv2.destroyWindow('Good Search')

# im = cv2.imread("dataset_final/images/elchainik.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('ЧАЙНИК', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')


# from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, build_model, StandardROIHeads
# import detectron2
# from detectron2.modeling.backbone import backbone
# from torch import nn
# from detectron2.config import get_cfg
#
#
# @BACKBONE_REGISTRY.register()
# class ToyBackbone(Backbone):
#   def __init__(self, cfg, input_shape):
#     super().__init__()
#     # create your own backbone
#     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)
#
#   def forward(self, image):
#     return {"conv1": self.conv1(image)}
#
#   def output_shape(self):
#     return {"conv1": ShapeSpec(channels=64, stride=16)}
#
# cfg = get_cfg()   # read a config
# cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # or set it in the config file
# model = build_model(cfg)  # it will find `ToyBackbone` defined above
#
# roi_heads = StandardROIHeads(
#   cfg, backbone.output_shape(),
#   # box_predictor=MyRCNNOutput(...)
# )


# # import matplotlib
# # from Cython import inline
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_teapot",
#                         {},
#                         "dataset/label/labels_my-project-name_2021-09-13-10-20-21.json",
#                         "path/dataset/images/*")
# filename = 'test.jpg'
# maskname = "dataset/label/labels_my-project-name_2021-09-13-10-20-21.json"
# # matplotlib inline
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
#
# img = plt.imread("/content/" + filename)
# plt.imshow(img)
# plt.axis('off')
# plt.show()
#
# plt.figure(figsize=(10,10))
# img = plt.imread("/content/output_demo/test/" + maskname)
# plt.imshow(img)
# plt.axis('off')
# plt.show()


# import time

# check pytorch installation:
# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.9")
# please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
# import detectron2

# im = cv2.imread("dataset_v2/images/depositphotos_1355596-stock-photo-blue-teapot.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')

# im = cv2.imread("dataset_v2/images/main2-8.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')

# im = cv2.imread("dataset_v2/images/unnamed.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')

# im = cv2.imread("dataset_v2/images/1040_a7944c61.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')
#
# im = cv2.imread("dataset_v2/images/1040_a7944c61.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')
#
# im = cv2.imread("dataset_v2/images/1040_a7944c61.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')
#
# im = cv2.imread("dataset_v2/images/1040_a7944c61.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Good Search', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('ЧАЙНИК')

### Train on a custom dataset ###
# Prepare the dataset #

### Нужно загрузить датасет и разархивировать в корень проекта ###

# !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
# !unzip balloon_dataset.zip > /dev/null

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# from detectron2.structures import BoxMode
#
#
# def get_balloon_dicts(img_dir):
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)
#
#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
#
#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]
#
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#
#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]
#
#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts
#
#
# def get_teapot_dicts(img_dir):
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)
#
#     dataset_dicts = []
#     test = imgs_anns['images']
#     # test = imgs_anns['images'].values()
#     test = {test}
#     for idx, v in enumerate(imgs_anns.values()):
#     # for idx, v in enumerate(imgs_anns['images'].values()):
#         record = {}
#
#         filename = os.path.join(img_dir, v["filename"])
#         # filename = os.path.join(img_dir, v["file_name"])
#         height, width = cv2.imread(filename).shape[:2]
#
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#
#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]
#
#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts
#
#
# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# # balloon_metadata = MetadataCatalog.get("balloon_train")
# balloon_metadata = MetadataCatalog.get("my_dataset_teapot")
#
#
# dataset_dicts = get_balloon_dicts("balloon_v0/train")
# # dataset_dicts = get_teapot_dicts("dataset_v2/images")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     # visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Balloon', out.get_image()[:, :, ::-1])
#     cv2.waitKey(1)
#     cv2.destroyWindow('Balloon')

### Train! ###

# from detectron2.engine import DefaultTrainer
#
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("balloon_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
#
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()


### Пока не разобрался с метрикой TensorBoard ###


# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

### Inference & evaluation using the trained model ###

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)
#
#
# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_balloon_dicts("balloon_v0/val")
# for d in random.sample(dataset_dicts, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=balloon_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('Visualize the prediction results', out.get_image()[:, :, ::-1])
#     cv2.waitKey(1)
#     cv2.destroyWindow('Visualize the prediction results')
#
#
#
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# #!!!Тут пока что ошибка
# evaluator = COCOEvaluator(dataset_name="balloon_val", output_dir=cfg.OUTPUT_DIR)
# val_loader = build_detection_test_loader(cfg, "balloon_val")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`
#
# ### Other types of builtin models ###
#
# # Inference with a keypoint detection model
# cfg = get_cfg()   # get a fresh new config
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Other types of builtin models', out.get_image()[:, :, ::-1])
# cv2.waitKey(1)
# cv2.destroyWindow('Other types of builtin models')
#
# # Inference with a panoptic segmentation model
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# predictor = DefaultPredictor(cfg)
# panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
# cv2.imshow('Inference with a panoptic segmentation model', out.get_image()[:, :, ::-1])
# cv2.waitKey(0)
# cv2.destroyWindow('Inference with a panoptic segmentation model')

### Run panoptic segmentation on a video ###

# This is the video we're going to process
# from IPython.display import YouTubeVideo, display
# video = YouTubeVideo("ll8TgCZ0plk", width=500)
# display(video)
