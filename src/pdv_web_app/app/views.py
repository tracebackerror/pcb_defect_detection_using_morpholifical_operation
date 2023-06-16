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

        messages.info(self.request, f'account done not exit plz sign in')
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
