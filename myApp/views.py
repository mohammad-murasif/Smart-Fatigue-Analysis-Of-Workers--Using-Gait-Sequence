from django.shortcuts import render
from django.shortcuts import render, redirect, resolve_url, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from django.shortcuts import render, HttpResponse, redirect
from .forms import Video_form, worker_registerForm
from .models import Video, worker
from django.contrib import messages
from myProject.settings import BASE_DIR1, BASE_DIR

from .util.pretreatment import videoFeed
from django.views.decorators import gzip
from .util.test import preProccessVideo
from .util.identification import WorkerRegistration, FaceRecognition
import threading
import uuid
from django.views.decorators.csrf import csrf_exempt

workerFaceRegistration = WorkerRegistration()
workerFaceRecognition = FaceRecognition()


def index(request):
    # last_video=Video.objects.last()

    # print(last_video.video.url)

    return render(request, 'myApp/home.html',)


# def home(request):
#     return render(request, 'home.html')


# register worker
def registerWorker(request):
    if request.method == 'POST':
        formWorker = worker_registerForm(request.POST)
        VideoForm = Video_form(request.POST, request.FILES)

        if VideoForm.is_valid() and formWorker.is_valid():
            VideoForm.save()
            formWorker.save()
            # Get last saved worker and Video
            WokerObj = worker.objects.last()
            VideoObj = Video.objects.last()
            videoPath = BASE_DIR + VideoObj.video.url
            workerFaceRegistration.register_worker(
                videoPath, WokerObj.workerId)
            print('SAVED')
        return redirect('results')
    else:
        formWorker = worker_registerForm()
        VideoForm = Video_form()

    context = {
        'formWorker': formWorker,
        'VideoForm': VideoForm,
    }

    return render(request, 'myApp/train.html', context)


def UpdateWorkerFaceData(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            workerid = request.POST.get('workerId')
            file = request.FILES['video']
            videoObj = Video(video=file)
            videoObj.save()
            videoPath = BASE_DIR + Video.objects.last().video.url
            result = workerFaceRecognition.update_worker_faceData(
                workerid, videoPath)
            if result == 0:
                return JsonResponse({'results': 'Facial Features Extracted Successfully'})
            else:
                return JsonResponse({'results': 'Face Not Detected'})
        except Exception as e:
            return JsonResponse({'results': e})
    else:
        form = Video_form()
        print(form)
        workers = worker.objects.all()
        return render(request, 'myApp/trainrecog.html', {'workers': workers})


def test(request):
    if request.method == 'POST':
        try:
            form = Video_form(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                return redirect('results')

        except Exception as e:
            return HttpResponse('<h1>Something went wrong..!</h1>')
    else:

        form = Video_form()
        return render(request, 'myApp/test.html', {'form': form})


processing_status = {}
identification_processing_status = {}

def resultstest(request):
    return render(request, 'myApp/resultstest.html')


def results(request):

    videoObj = Video.objects.last()
    path = BASE_DIR + videoObj.video.url
    task_id = str(uuid.uuid4())
    task1_id = str(uuid.uuid4())

    # Store initial status
    processing_status[task_id] = 'Processing'
    identification_processing_status[task1_id] = 'Identfication-in-process'

    # Start the processing in a new thread
    thread = threading.Thread(target=process_fatigue, args=(task_id, path))
    thread.start()
    thread1=threading.Thread(target=process_identification, args=(task1_id, path))
    thread1.start()
    # if request.method=='POST':

    #     res=workerFaceRecognition.faceDetectAndRecognize(path)
    #     if res =='NOTDETECTED':
    #         return JsonResponse({'workerName':res})
    #     else:
    #         workerobj=get_object_or_404(worker,workerId=res)
    #         print(workerobj.workerName)
    #         return JsonResponse({'workerName':workerobj.workerName})

    # res=Fatigueresult(path)
    # print(res)
    # result = preProccessVideo(path)
    # print(result)

    return render(request, 'myApp/results.html', {'task_id': task_id, 'status': 'Processing','task1_id':task1_id,'status1':'Identfication-in-process'})


def process_fatigue(task_id, path):
    try:
        # workerFaceRecognition.train_classifier()
        res = preProccessVideo(path)
        # workerId=workerFaceRecognition.faceDetectAndRecognize(path)
        processing_status[task_id] = res
        # if workerId:
            # workerObj=get_object_or_404(worker,id=workerId)
            # processing_status['workername'] = workerObj.workerName

    except Exception as e:
        processing_status[task_id] = f'Error: {str(e)}'


def get_status(request, task_id):
    status = processing_status.get(task_id, 'No such task')
    if status == 'Processing':
        return JsonResponse({'status': 'Processing'})
    elif isinstance(status, int) and status == 0:
        return JsonResponse({'status': '0'})
    elif isinstance(status, int) and status == 1:
        return JsonResponse({'status': '1'})
    else:
        return JsonResponse({'status': status})

def process_identification(task1_id, path):
    try:
        res = workerFaceRecognition.faceDetectAndRecognize(path)
        print(f'IM in proccess identfication:{res}')
        identification_processing_status[task1_id] = res
    except Exception as e:
        identification_processing_status[task1_id] = f'Error: {str(e)}'

def get_status_identfication(request, task1_id):
    status1=identification_processing_status.get(task1_id,'No such task')
    if status1 == 'Identfication-in-process':
        return JsonResponse({'status1':'Identfication-in-process'})
    else:
            workerobj = get_object_or_404(worker, workerId=status1)
            print(f'in get status:{workerobj.workerName}')
            return JsonResponse({'workerId': workerobj.workerId, 'workerName': workerobj.workerName})





def identifyWorkerView(request):
    if request.method == 'GET':
        videoObj = Video.objects.last()
        path = BASE_DIR + videoObj.video.url
        res = workerFaceRecognition.faceDetectAndRecognize(path)
        if res == 'NOTDETECTED':
            return JsonResponse({'workerName': 'NOTDETECTED'})
        else:
            workerobj = get_object_or_404(worker, workerId=res)
            return JsonResponse({'workerId': workerobj.workerId, 'workerName': workerobj.workerName})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)



        
    
        
@csrf_exempt
def trainWorkerFaces(request):
    if request.method == 'POST':
        try:
            workerFaceRecognition.train_classifier()
            return JsonResponse({'message': 'Successfully trained'})
        except Exception as e:
            return JsonResponse({'message': 'Training failed', 'error': str(e)}, status=500)
    return JsonResponse({'message': 'Invalid request method'}, status=405)


    

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(videoFeed(), content_type="multipart/x-mixed-replace;boundary=frame")




