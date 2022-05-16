from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.views.decorators.csrf import csrf_exempt, csrf_protect

import numpy as np
import joblib


# Create your views here.
@csrf_exempt
@api_view(['GET', 'POST'])
def home_page(request):
    if request.method == 'GET':
        return render(request, 'Inputs.html')

    if request.method == 'POST':
        data = request.data
        try:
            model = open("./api/modelo.pkl", "rb")
            array_demo = np.array([float(data['var_1']), float(data['var_2']), float(data['var_3']), float(data['var_4']), float(data['var_5'])])
            preds = array_demo.reshape(1, -1)
            tree_model = joblib.load(model)
            result = tree_model.predict(preds)

            if result[0] == 'Graduate':
                return render(request, 'graduate.html')
            else:
                return render(request, 'dropout.html')

        except ValueError:
            error_message = 'Una de las variables se encuentra vacia'

            return Response(error_message)
