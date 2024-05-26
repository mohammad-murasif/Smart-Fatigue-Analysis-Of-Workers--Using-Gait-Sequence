import cv2
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import os
import collections
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import glob
from myProject.settings import BASE_DIR1,BASE_DIR
from myApp.models import Video,worker
import joblib
model_path=BASE_DIR + '/myApp/util/model/recognitionmodel.pkl'
class WorkerRegistration:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def register_worker(self, path, worker_id):
        print(f"Processing video for worker ID: {worker_id}")
        cam = cv2.VideoCapture(path)
        count = 0

        embeddings = []

        while True:
            ret, img = cam.read()
            if not ret:
                print("VIDEO NOT LOADED")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                embedding = self.get_embedding(face_img)
                embeddings.append(embedding)

                count += 1
                if count >= 30:
                    break

            if count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save the embeddings for this worker
        if len(embeddings) !=0:
            embeddings = np.vstack(embeddings)
            embeddingPath= BASE_DIR + '/media/embeddings'
            os.makedirs(embeddingPath, exist_ok=True)
            np.save(f'{embeddingPath}/worker_{worker_id}.npy', embeddings)
            print(f"Saved embeddings for worker ID: {worker_id}")
        else:
            lastVideo=Video.objects.last()
            lastWorker= worker.objects.last()
            lastVideo.delete
            print("Video Not Suitable for Face Detection")
            
            

    def get_embedding(self, face_img):
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img / 255.0  # Normalize to [0, 1]
        face_img = np.transpose(face_img, (2, 0, 1))  # HWC to CHW
        face_img = torch.tensor(face_img, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(face_img).numpy()
        return embedding


class FaceRecognition:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.classifier = SVC(kernel='linear', probability=True)
        self.label_encoder = LabelEncoder()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def train_classifier(self):
        X, y = self.load_all_embeddings()
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("The number of classes has to be greater than one; got %d class" % len(unique_classes))
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.classifier.fit(X, y_encoded)
        print("OK IN CLASSIFIER DONE")

    def load_all_embeddings(self):
        embeddings = []
        labels = []
        embeddingsPath= BASE_DIR + '/media/embeddings/worker_*.npy'
        for filepath in glob.glob(embeddingsPath):
            worker_id = int(filepath.split('_')[1].split('.')[0])
            worker_embeddings = np.load(filepath)
            embeddings.append(worker_embeddings)
            labels += [worker_id] * worker_embeddings.shape[0]
        return np.vstack(embeddings), np.array(labels)

    def faceDetectAndRecognize(self, path):
        cam = cv2.VideoCapture(path)
        predictions = []
        embeddings = []
        self.train_classifier()
        face_count=0

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            if face_count >50:
                break


            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                embedding = self.get_embedding(face_img)

                # Predict using the classifier
                embedding = embedding.reshape(1, -1)  # Reshape for prediction
                prediction = self.classifier.predict(embedding)
                predicted_worker_id = self.label_encoder.inverse_transform(prediction)
                predictions.append(predicted_worker_id[0])
                embeddings.append(embedding.flatten())

                # Draw a rectangle around the face and display the ID
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, str(predicted_worker_id[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                face_count +=1

            print(f'TOTAL FACES ::::: {face_count}')
            
            # cv2.imshow('Video', img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Find the most frequent worker ID and check its occurrence
        if predictions:
            most_common_worker_id, count = collections.Counter(predictions).most_common(1)[0]
            if count > 5:
                print(f'Most frequent worker ID: {most_common_worker_id}')
                return most_common_worker_id
            else:
                print('No face detected')
                return "No face detected"
        else:
            print('No faces detected')
            return "NOTDETECTED"

    def get_embedding(self, face_img):
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img / 255.0  # Normalize to [0, 1]
        face_img = np.transpose(face_img, (2, 0, 1))  # HWC to CHW
        face_img = torch.tensor(face_img, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(face_img).numpy()
        return embedding

    def update_worker_faceData(self, worker_id, video_path):
        # most_frequent_worker_id, new_embeddings = self.faceDetectAndRecognize(video_path)
        
        
        # if most_frequent_worker_id == "No face detected":
        #     print('No sufficient face data to register the worker.')
        #     return
        print(f"Processing video for worker ID: {worker_id}")
        cam = cv2.VideoCapture(video_path)
        print(video_path)
        count = 0

        embeddings = []

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                embedding = self.get_embedding(face_img)
                embeddings.append(embedding)

                count += 1
                if count >= 30:
                    break

            if count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()
        
        embedding_path = f'{BASE_DIR}/media/embeddings/worker_{worker_id}.npy'
        print(embeddings)
        if len(embeddings)!=0:
            if os.path.exists(embedding_path) and len(embeddings) !=0:
                # Load existing embeddings and append new ones
                embeddings = np.vstack(embeddings)
                existing_embeddings = np.load(embedding_path)
                combined_embeddings = np.vstack((existing_embeddings, embeddings))
            else:
                # Use new embeddings
                combined_embeddings = embeddings
        
            # Save combined embeddings
            np.save(embedding_path, combined_embeddings)
            print(f'Worker {worker_id} updated with {combined_embeddings.shape[0]} embeddings.')
            return 0
        else:
            return 1

    def check_data(self):
        X, y = self.load_all_embeddings()
        print(f'Embeddings shape: {X.shape}')
        print(f'Labels shape: {y.shape}')
        print(f'Unique classes: {np.unique(y)}')
        print(f'Class counts: {np.bincount(y)}')
