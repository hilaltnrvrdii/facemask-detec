
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# çerçevenin boyutlarını al ve ordan bir blob oluştur
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	#yüz listesini, karşılığında gelen konumlarını ve yüz maskesi ağımızdan tahminler listesi

	faces = []
	locs = []
	preds = []

	# tespit edilenler üzerinde döngü
	for i in range(0, detections.shape[2]):
		# algılama ile ilişkili olasılığı çıkar
		confidence = detections[0, 0, i, 2]

		# doğru çalıştığından emin olmak ve zayıf algılamaları filtrelemek için if şartı
		if confidence > 0.5:
			#yüz tanıma kutucuğunun (x-y) koordinatlarını hesaplama
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#çerçevenin sınırlayıcı kutunun boyutları dahilinde olduğundan emin olalım
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# yüz ROI'sini çıkarıp BGR'dan RGB'ye dönüştür
			# 224x224 ölçüsünde boyutlandıralım
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# listelere yüz ve sınırlayıcı kutularını ekle
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# en az bir yüz bulunması durumunda maske tespiti yap
	if len(faces) > 0:
		# daha hızlı sonuç alabilmek için yukarıdaki for döngüsündeki
		# tek tek tahminler yerine yüzler üzerinde
		# aynı anda toplu tahminler yapacağız
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# yüz konumlarının iki koordinatını ve bunlara karşılık gelen konumları döndür
	return (locs, preds)

# serileştirilmiş yüz dedektör modelini yükle
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# yüz maskesi dedektör modelini yükle
maskNet = load_model("mask_detector.model")

# video akışını başlat
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# video akışındaki kareler üzerinde döngü
while True:
	# görüntülenen video akışından çerçeveyi al ve max 400 pixel genişliğe
	#sahip olacak şekilde yeniden boyutlandır
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	#çerçevedeki yüzleri algıla ve yüz maskesi takıp takmadığını belirle
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# algılanan yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü
	for (box, pred) in zip(locs, preds):
		# sınırlayıcı kutuyu ve tahminleri aç
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		#sınırlayıcı kutuyu ve metni çizmek için kullanılacak sınıf etiketini
		#ve rengini belirle
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		#etikete olasılığı dahil et
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# etiket ve sınırlayıcı kutu dikdörtgenini çıktı çerçevesinde görüntüle
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	#çıktı çerçevesini göster
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' tuşuna basıldığında döngüden çık
	if key == ord("q"):
		break

# pencereleri temizle
cv2.destroyAllWindows()
vs.stop()
