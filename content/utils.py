import numpy as np
import cv2

def generate_colors(num_classes):
	"""Generate a list of unique colors for the given number of classes."""
	np.random.seed(42)  # For reproducibility
	colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
	return colors

def display_detections(model, frame, results, colors):
	"""Draw bounding boxes and labels on the frame."""
	for result in results:
		boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy (x1, y1, x2, y2)
		scores = result.boxes.conf.cpu().numpy()  # Confidence scores
		labels = result.boxes.cls.cpu().numpy()  # Class IDs
		
		for box, score, label in zip(boxes, scores, labels):
			x1, y1, x2, y2 = map(int, box)
			
			# Get the color for the current class
			color = tuple(int(c) for c in colors[int(label)])
			
			# Draw the bounding box
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			
			# Display label and confidence score
			text = f"{model.names[int(label)]}: {score:.2f}"
			cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return frame

def display_masks(model, frame, results, class_colors, alpha=0.3):
	"""Draw bounding boxes, labels, and masks on the frame."""
	overlay = np.zeros_like(frame, dtype=np.uint8)

	for result in results:
		boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy (x1, y1, x2, y2)
		scores = result.boxes.conf.cpu().numpy()  # Confidence scores
		labels = result.boxes.cls.cpu().numpy()  # Class IDs


		if hasattr(result, 'masks') and result.masks is not None:
			masks = result.masks.data.cpu().numpy()  # Extract masks
			for mask, label in zip(masks, labels):
				color = tuple(int(c) for c in class_colors[int(label)])

				# Resize mask to match frame size if needed
				mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
				mask = (mask > 0.5).astype(np.uint8)  # Binarize mask

				# Apply the mask with transparency
				overlay[mask == 1] = color

		
		for box, score, label in zip(boxes, scores, labels):
			x1, y1, x2, y2 = map(int, box)

			# Get the color for the current class
			color = tuple(int(c) for c in class_colors[int(label)])

			# Draw the bounding box
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

			# Display label and confidence score
			text = f"{model.names[int(label)]}: {score:.2f}"
			cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)


	return frame
