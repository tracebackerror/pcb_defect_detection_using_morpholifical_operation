import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from skimage.metrics import structural_similarity as compare_ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages


from .forms import PCBForm
from .image_ssim import pcb_defect_detection_algorithm

from django.views.generic.base import TemplateView
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from django.http import JsonResponse

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from skimage.metrics import structural_similarity as compare_ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


def get_image_pairs(directory):
    """
    Dynamically scan the given directory and form IMAGE_PAIRS by matching
    'original' and 'defective' image files based on their base sample names.

    Naming Convention:
      - Original images: samplename_original.jpg
      - Defective images: samplename_defecttype_defective.jpg

    Args:
        directory (str): Path containing the images.

    Returns:
        list: A list of tuples [(original_path, defective_path), ...].
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # File containers
    original_files = {}
    defective_files = {}

    # Regex patterns to extract sample name and file type
    original_pattern = re.compile(r"^(sample\d+|pcb\d+)_original\.jpg$", re.IGNORECASE)
    defective_pattern = re.compile(
        r"^(sample\d+|pcb\d+)_.*?_defective\.jpg$", re.IGNORECASE
    )

    # Find all .jpg files in the given directory
    files = [file for file in sorted(os.listdir(directory)) if file.endswith(".jpg")]

    # Loop through files and match based on pattern
    for filename in files:
        original_match = original_pattern.match(filename)
        defective_match = defective_pattern.match(filename)

        if original_match:
            # Extract sample name (e.g., pcb1_original -> pcb1)
            sample_name = original_match.group(1)
            original_files[sample_name] = os.path.join(directory, filename)

        elif defective_match:
            # Extract sample name (e.g., pcb1_missingpinhole_defective -> pcb1)
            sample_name = defective_match.group(1)
            defective_files.setdefault(sample_name, []).append(
                os.path.join(directory, filename)
            )

    # Pair up original and defective images based on sample name
    image_pairs = []
    for sample_name, original_path in original_files.items():
        if sample_name in defective_files:
            # For each original, find all the defective variants
            for defective_path in defective_files[sample_name]:
                image_pairs.append((original_path, defective_path))

    if not image_pairs:
        raise ValueError("No valid image pairs found in the directory.")

    return image_pairs


@csrf_exempt
def performance_check(request):
    """
    Backend view that compares the performance of SSIM and CNN-based defect detection algorithms.
    """
    try:
        # Path to test images directory
        TEST_IMAGE_DIR = os.path.join(
        settings.BASE_DIR, 'app', 'tests_performance'
        )
        # Dynamically load image pairs
        IMAGE_PAIRS = get_image_pairs(TEST_IMAGE_DIR)
        if not IMAGE_PAIRS:
            return JsonResponse({'status': 'error', 'message': 'No image pairs found in the directory.'})

        # Load pre-trained VGG16 model
        cnn_model = create_pretrained_vgg16_for_pcb()

        # Initialize performance metrics
        results_ssim = []
        results_cnn = []

        # Store PCB pair labels for charting
        pcb_labels = []

        # Iterate through all image pairs
        for original_path, defective_path in IMAGE_PAIRS:
            # Read images
            original_image = cv2.imread(original_path)
            defective_image = cv2.imread(defective_path)

            # Extract PCB Label (Sample Name)
            pcb_name = os.path.basename(original_path).split("_")[0]
            pcb_labels.append(pcb_name)

            # SSIM Performance
            ssim_score = compute_ssim_metrics(original_image, defective_image)
            results_ssim.append(float(ssim_score))  # Ensure it's serializable

            # CNN-Based Performance
            cnn_original, cnn_defective = predict_cnn_metrics(cnn_model, original_image, defective_image)
            cnn_difference = abs(cnn_original - cnn_defective)
            results_cnn.append(float(cnn_difference))  # Ensure it's serializable

        # Return JSON response for Highcharts
        return JsonResponse({
            'status': 'success',
            'message': 'Performance metrics computed successfully!',
            'data': {
                'labels': pcb_labels,  # PCB Pair Labels (x-axis)
                'ssim_scores': results_ssim,  # SSIM Scores (bar 1)
                'cnn_scores': results_cnn  # CNN Prediction Differences (bar 2)
            }
        })

    except Exception as e:
        # Return error details to the frontend
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def compute_ssim_metrics(original, defective):
    """
    Compute SSIM (Structural Similarity Index Metric) between two images.
    """
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    defective_gray = cv2.cvtColor(defective, cv2.COLOR_BGR2GRAY)

    # Compute SSIM and return score
    ssim_score, _ = compare_ssim(original_gray, defective_gray, full=True)
    return ssim_score


def predict_cnn_metrics(model, original, defective):
    """
    Use a pre-trained CNN model to predict similarity or classify.
    """
    # Preprocess input images for the CNN
    original_resized = cv2.resize(original, (224, 224))
    defective_resized = cv2.resize(defective, (224, 224))

    # Convert the image to the CNN input format
    original_array = img_to_array(original_resized)
    defective_array = img_to_array(defective_resized)

    original_array = preprocess_input(np.expand_dims(original_array, axis=0))
    defective_array = preprocess_input(np.expand_dims(defective_array, axis=0))

    # Get predictions (assuming binary classification: Defective/Non-Defective)
    original_prediction = model.predict(original_array)[0][0]
    defective_prediction = model.predict(defective_array)[0][0]

    return original_prediction, defective_prediction


def plot_performance_graph(ssim_scores, cnn_scores):
    """
    Plot a bar graph comparing SSIM and CNN model performance across PCB pairs.
    """
    x_labels = [f"PCB Pair {i + 1}" for i in range(len(ssim_scores))]

    # Extract CNN scores
    cnn_scores_processed = [np.abs(cnn_orig - cnn_def) for cnn_orig, cnn_def in cnn_scores]

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_labels))

    plt.bar(indices - 0.2, ssim_scores, width=0.4, label="SSIM Scores", color="blue")
    plt.bar(indices + 0.2, cnn_scores_processed, width=0.4, label="CNN Difference", color="green")

    plt.xlabel("PCB Pairs")
    plt.ylabel("Score")
    plt.title("Performance Comparison: SSIM vs. CNN")
    plt.xticks(indices, x_labels)
    plt.legend(loc="upper right")

    # Save the plotted graph
    graph_path = os.path.join("static", "performance_comparison.png")
    plt.savefig(graph_path)
    plt.close()

def create_pretrained_vgg16_for_pcb():
    """
    Load a pre-trained VGG16 model and adapt it for binary classification (Defective/Non-Defective).
    """
    # Load the pre-trained VGG16 model with ImageNet weights, exclude top classifier layers
    weights_path = os.path.join(
        settings.BASE_DIR, 'app', 'models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )

    base_model = VGG16(
        weights=weights_path,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Add custom layers on top for binary classification
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (defective/non-defective)

    # Freeze the convolutional layers of the pre-trained model so they are not updated during training
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


@csrf_exempt
def process_cnn(request):
    """
    Process PCB images with a pre-trained VGG16 model to classify them as defective or non-defective.
    """
    if request.method == 'POST':
        try:
            # Load the pre-trained model
            model = create_pretrained_vgg16_for_pcb()

            # Paths to input images
            source_image_path = "./static/first_image.jpg"
            destination_image_path = "./static/second_image.jpg"

            # Detect defects in the images
            results = detect_defects_with_cnn(model, source_image_path, destination_image_path)

            return JsonResponse({
                'status': 'success',
                'message': 'Image processing using VGG16 completed.',
                'results': results
            })

        except Exception as e:
            print(e)
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)


def detect_defects_with_cnn(model, source_image_path, destination_image_path):
    """
    Detect defects in input images using a pre-trained VGG16 model.

    :param model: Pre-trained model for defect detection.
    :param source_image_path: Path to the first input image (reference PCB image).
    :param destination_image_path: Path to the second input image (PCB to inspect).
    :return: Dictionary containing classification results for both images.
    """
    # Preprocess source image
    source_image = cv2.imread(source_image_path)
    source_image_resized = cv2.resize(source_image, (224, 224))
    source_array = img_to_array(source_image_resized)
    source_array = preprocess_input(source_array)
    source_array = np.expand_dims(source_array, axis=0)

    # Preprocess destination image
    destination_image = cv2.imread(destination_image_path)
    destination_image_resized = cv2.resize(destination_image, (224, 224))
    destination_array = img_to_array(destination_image_resized)
    destination_array = preprocess_input(destination_array)
    destination_array = np.expand_dims(destination_array, axis=0)

    # Perform predictions
    source_prediction = model.predict(source_array)[0][0]  # Sigmoid output (probability of being defective)
    destination_prediction = model.predict(destination_array)[0][0]

    # Threshold to decide defective vs non-defective (probability > 0.5 = defective)
    threshold = 0.5
    results = {
        "source_image": "Defective" if source_prediction > threshold else "Non-Defective",
        "destination_image": "Defective" if destination_prediction > threshold else "Non-Defective",
        "source_probability": float(source_prediction),
        "destination_probability": float(destination_prediction),
    }

    # Save annotated results for visualization
    annotate_and_save_images(source_image, destination_image, results)

    return results


def annotate_and_save_images(source_image, destination_image, results):
    """
    Annotate images with defect classification results and save them.

    :param source_image: Image array of the source (reference) PCB.
    :param destination_image: Image array of the destination PCB.
    :param results: Classification results.
    """
    # Annotate source image
    label = f"Source: {results['source_image']} (P: {results['source_probability']:.2f})"
    source_annotated = cv2.putText(
        source_image.copy(), label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if results['source_image'] == "Non-Defective" else (0, 0, 255), 2
    )
    cv2.imwrite("./static/vgg16_source_result.jpg", source_annotated)

    # Annotate destination image
    label = f"Destination: {results['destination_image']} (P: {results['destination_probability']:.2f})"
    destination_annotated = cv2.putText(
        destination_image.copy(), label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if results['destination_image'] == "Non-Defective" else (0, 0, 255), 2
    )
    cv2.imwrite("./static/vgg16_destination_result.jpg", destination_annotated)

def logout_view(request):
    logout(request)
    return redirect(reverse("index"))


def register_view(request):
    username = request.POST.get('username', None)
    password = request.POST.get('password', None)
    user = User.objects.create_user(username, "", password)
    user = authenticate(username=username, password=password)

    if user is not None:
        # A backend authenticated the credentials
        form = login(request, user)
        messages.success(request, f' welcome {username} !!')
        return redirect(reverse("index"))
    return redirect(reverse("register"))


class LoginView(TemplateView):
    template_name = 'login.html'

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)

        username = self.request.POST.get('username', None)
        password = self.request.POST.get('password', None)

        user = authenticate(username=username, password=password)
        #import pdb; pdb.set_trace()
        if user is not None:
            # A backend authenticated the credentials
            form = login(self.request, user)
            messages.success(self.request, f' welcome {username} !!')
            return redirect(reverse("index"))

        messages.info(self.request, f'Incorrect Credentials')
        return self.render_to_response(context)

@login_required(login_url="/login")
def index(request):
    template = loader.get_template('index.html')


    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = PCBForm(request.POST, request.FILES)
        # check whether it's valid:
        #
        if form.is_valid():
            context = {
                'form': form,
            }
            pcb_defect_detection_algorithm(request.FILES['source_image'], request.FILES['destination_image'])
            return HttpResponse(template.render(context, request))

    else:
        form = PCBForm()

    context = {
        'form': form,
    }

    return HttpResponse(template.render(context, request))
