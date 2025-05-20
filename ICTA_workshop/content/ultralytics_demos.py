import cv2
import numpy as np
from ultralytics import YOLO, FastSAM, SAM, YOLOE
from ultralytics.models.fastsam import FastSAMPredictor
import torch
from ultralytics.utils.plotting import Colors
import argparse
from utils import *

# CUDA_VISIBLE_DEVICES=1 python ultralytics_demos.py


# data_dir = '/data/ICTA_workshop/pexel_videos/'
data_dir = '/data/ICTA_workshop/data/videos/'

f = 0.5

def predict_yolo():
	cam = cv2.VideoCapture(f'{data_dir}/office.mp4')
	model = YOLO("yolo12n.pt")

	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

	class_colors = generate_colors(len(model.names))

	while True:
		ret, frame = cam.read()
		results = model.predict(frame)

		display_detections(model, frame, results, class_colors)

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

def predict_yolo_seg():

	cam = cv2.VideoCapture(f'{data_dir}/office.mp4')
	model = YOLO("yolo11s-seg.pt")

	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

	class_colors = generate_colors(len(model.names))

	while True:
		ret, frame = cam.read()
		frame = cv2.resize(frame, None, fx=f, fy=f)
		results = model.predict(frame)
		frame = display_masks(model, frame, results, class_colors)

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()



def predict_yolo_world():
	cam = cv2.VideoCapture(f'{data_dir}/office.mp4')

	cv2.namedWindow('YOLO Predictions', cv2.WINDOW_NORMAL)

	class_colors = generate_colors(50)

	model = YOLO("yolov8l-worldv2.pt")

	prompts = [
		"phone",
		"mug",
		"coffee",
		"pen"
	]

	# prompts = [
	# 	"vest",
	# 	"visibility",
	# 	"high-visibility vest"
	# ]

	# prompts = [
	# 	'red-haired woman',
	# 	'man in blue shirt'
	# ]

	model.set_classes(prompts)

	while True:
		ret, frame = cam.read()
		if not ret:
			break

		results = model.predict(frame)

		frame = display_detections(model, frame, results, class_colors)

		cv2.imshow("YOLO Predictions", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

def predict_yolo_pose():

	cam = cv2.VideoCapture(f'{data_dir}/forest.mp4')

	model = YOLO("yolo11s-pose.pt")

	class_colors = generate_colors(len(model.names))
	f = 0.25

	while True:
		ret, frame = cam.read()
		frame = cv2.resize(frame, None, fx=f, fy=f)		

		results = model.predict(frame, show=True)

	cam.release()

def predict_fastsam():

	cam = cv2.VideoCapture(f'{data_dir}/office.mp4')

	model = FastSAM("FastSAM-s.pt")
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

	class_colors = generate_colors(50)
	f = 0.25

	while True:
		ret, frame = cam.read()

		if not ret:
			break
		
		frame = cv2.resize(frame, None, fx=f, fy=f)


		# results = model(frame, texts="coffe mug")
		# results = model(frame, texts="person on the left")
		results = model(frame, texts="red-haired woman.man in blue shirt")

		frame = display_detections(model, frame, results, class_colors)

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

def predict_yoloe():

	# data_dir = 'data/videos/'	

	# cam = cv2.VideoCapture(0)
	# cam = cv2.VideoCapture(f'{data_dir}/prec.mp4')
	cam = cv2.VideoCapture(f'{data_dir}/office.mp4')
	# cam = cv2.VideoCapture(f'{data_dir}/forest.mp4')

	model = YOLOE("yoloe-11l-seg.pt")
	# model = YOLOE("yoloe-11l-seg-pf.pt")
	# names = ["person", "wall"]
	names = [
		"phone",
		"mug",
		"coffee",
		"pen"
	]
	# names = [
	# 	'red-haired woman',
	# 	'man in blue shirt'
	# ]
	model.set_classes(names, model.get_text_pe(names))

	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

	class_colors = generate_colors(len(model.names))

	while True:
		ret, frame = cam.read()

		results = model.predict(frame)

		frame = display_detections(model, frame, results, class_colors)

		cv2.imshow("frame", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		

	cam.release()
	# cv2.destroyAllWindows()

if __name__=='__main__':

	# Instantiate the parser
	parser = argparse.ArgumentParser(description='Optional app description')

	parser.add_argument('task', type=str, help='One of the tasks must be provided: "detect", "segment", "pose", "world", "SAM", "yoloe"')

	args = parser.parse_args()

	if args.task==None or args.task=='detect':
		predict_yolo()
	elif args.task=='segment':
		predict_yolo_seg()
	elif args.task=='world':
		predict_yolo_world()
	elif args.task=='pose':
		predict_yolo_pose()
	elif args.task=='SAM':
		predict_fastsam()
	elif args.task=='yoloe':
		predict_yoloe()
	else:
		raise Exception