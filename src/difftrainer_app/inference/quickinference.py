from difftrainer_app.modules.device import pick_device
import subprocess, os, yaml, sys, json
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ezlocalizr import ezlocalizr
from difftrainer_app.paths import ASSETS_ROOT, REPOSITORY_ROOT, STRINGS_ROOT, ensure_user_config

ctk.set_default_color_theme(str(ASSETS_ROOT / 'ds_gui.json'))
main_path = str(REPOSITORY_ROOT)
diffolder = os.path.join(main_path, "DiffSinger")
ckpts = os.path.join(diffolder, "checkpoints")
os.chdir(main_path)
realpython = sys.executable
inference_device = pick_device()
print(f"Available compute device: {inference_device}")

guisettings = {
    'lang': 'pt_BR',
}


settingspath = ensure_user_config()
if settingspath.exists():
    with open(settingspath, 'r', encoding='utf-8') as c:
        try:
                guisettings.update(yaml.safe_load(c))
                c.close()
        except yaml.YAMLError as exc:
            print("No settings detected, usando Português (Brasil) como idioma padrão")

ctk.FontManager.load_font(os.path.join(str(ASSETS_ROOT),"RedHatDisplay-Regular.ttf"))
ctk.FontManager.load_font(os.path.join(str(ASSETS_ROOT),"MPLUS2-Regular.ttf"))
ctk.FontManager.load_font(os.path.join(str(ASSETS_ROOT),"NotoSansSC-Regular.ttf"))
ctk.FontManager.load_font(os.path.join(str(ASSETS_ROOT),"NotoSansTC-Regular.ttf"))

font_en = 'Red Hat Display'
font_jp = 'M PLUS 2'
font_cn = 'Noto Sans SC'
font_tw = 'Noto Sans TC'

varckptQ = ''

def var():
        varget = filedialog.askdirectory(title=L('getvar'), initialdir = "DiffSinger/checkpoints")
        os.chdir(main_path)
        varckpt = os.path.relpath(varget, ckpts)
        print(varckpt)
        global varckptQ
        varckptQ = varckpt
def aco():
        acoget = filedialog.askdirectory(title=L('getaco'), initialdir = "DiffSinger/checkpoints")
        os.chdir(main_path)
        acockpt = os.path.relpath(acoget, ckpts)
        print(acockpt)
        aco_config = acoget + "/config.yaml"
        with open(aco_config, "r", encoding = "utf-8") as config:
                conf = yaml.safe_load(config)
        global bre_status
        bre_status = conf["use_breathiness_embed"]
        global ene_status
        ene_status = conf["use_energy_embed"]
        global ten_status
        ten_status = conf["use_tension_embed"]
        global voc_status
        voc_status = conf["use_voicing_embed"]
        global acockptQ
        acockptQ = acockpt
def ds():
        dsinput = filedialog.askopenfilename(title=L('getds'), filetypes=[(L('select_ds_files'), "*.ds")])
        print(dsinput)
        print(L('eval'))
        with open(dsinput, 'r', encoding='utf-8') as file:
             data = json.load(file)

        global hasdur
        def find_dur(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'ph_dur':
                        return True
                    else:
                        if find_dur(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_dur(item):
                        return True
            return False
        if find_dur(data):
             hasdur = True
             print(L('hasdur'))
        else:
             hasdur = False

        global haspitch
        def find_pitch(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'f0_seq':
                        return True
                    else:
                        if find_pitch(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_pitch(item):
                        return True
            return False
        if find_pitch(data):
             haspitch = True
             print(L('haspitch'))
        else:
             haspitch = False

        global hastension
        def find_tension(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'tension':
                        return True
                    else:
                        if find_tension(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_tension(item):
                        return True
            return False
        if find_tension(data):
             hastension = True
             print(L('hasten'))
        else:
             hastension = False

        global hasvoicing
        def find_voicing(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'voicing':
                        return True
                    else:
                        if find_voicing(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_voicing(item):
                        return True
            return False
        if find_voicing(data):
             hasvoicing = True
             print(L('hasvoc'))
        else:
             hasvoicing = False

        global hasenergy
        def find_energy(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'energy':
                        return True
                    else:
                        if find_energy(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_energy(item):
                        return True
            return False
        if find_energy(data):
             hasenergy = True
             print(L('hasene'))
        else:
             hasenergy = False

        global hasbreathiness
        def find_breathiness(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'breathiness':
                        return True
                    else:
                        if find_breathiness(value):
                            return True
            if isinstance(data, list):
                for item in data:
                    if find_breathiness(item):
                        return True
            return False
        if find_breathiness(data):
             hasbreathiness = True
             print(L('hasbre'))
        else:
             hasbreathiness = False

        print(L('eval2'))
        global dsinputQ
        dsinputQ = dsinput
        dsloc = os.path.dirname(dsinput)
        global dslocQ
        dslocQ = dsloc
        dsname = os.path.basename(dsinput)
        dsname2 = os.path.splitext(dsname)[0]
        global dsname2Q
        dsname2Q = dsname2
        postvar = dsname2 + "_var"
        global postvarQ
        postvarQ = postvar
        postvards = dsloc + "/" + postvar + ".ds"
        global postvardsQ
        postvardsQ = postvards
        global renderedQ
        rendered = os.path.join(dsloc, dsname2) + ".wav"
        renderedQ = rendered


def is_macos():
      return sys.platform == "darwin"


def is_windows():
      return sys.platform.startswith("win")


def play_audio(audio_path):
      if is_macos():
            subprocess.run(["afplay", audio_path], check=True)
      elif is_windows():
            escaped_path = audio_path.replace("'", "''")
            powershell = f"$player = New-Object System.Media.SoundPlayer -ArgumentList '{escaped_path}'; $player.PlaySync()"
            subprocess.run(["powershell", "-NoProfile", "-Command", powershell], check=True)
      else:
            subprocess.run(["xdg-open", audio_path], check=True)


def render():
      spkname = spk.get()
      langname = lcode.get()
      pdur = overdur.get()
      ppit = overpit.get()
      pbre = overbre.get()
      prene = overene.get()
      pten = overten.get()
      pvoc = overvoc.get()
      inference_env = os.environ.copy()
      inference_env["PYTHONPATH"] = os.pathsep.join(
            path for path in (main_path, str(REPOSITORY_ROOT / "src"), diffolder, inference_env.get("PYTHONPATH", "")) if path
      )
      if varckptQ != '':
            cmd1 = [realpython, 'scripts/infer.py', 'variance', dsinputQ, '--exp', varckptQ, '--spk', spkname, '--lang', langname, '--out', dslocQ, '--title', postvarQ]
            if pdur == True:
                 cmd1.append('--predict')
                 cmd1.append('dur')
            if ppit == True:
                 cmd1.append('--predict')
                 cmd1.append('pitch')
            if pbre == True:
                 cmd1.append('--predict')
                 cmd1.append('breathiness')
            if prene == True:
                 cmd1.append('--predict')
                 cmd1.append('energy')
            if pten == True:
                 cmd1.append('--predict')
                 cmd1.append('tension')
            if pvoc == True:
                 cmd1.append('--predict')
                 cmd1.append('voicing')
            if (pdur == True or ppit == True or prene == True or pten == True or pvoc == True) and pbre == False and bre_status == True and hasbreathiness == False:
                 cmd1.append('--predict')
                 cmd1.append('breathiness')
            if (pdur == True or ppit == True or pbre == True or pten == True or pvoc == True) and prene == False and ene_status == True and hasenergy == False:
                 cmd1.append('--predict')
                 cmd1.append('energy')
            if (pdur == True or ppit == True or pbre == True or prene == True or pvoc == True) and pten == False and ten_status == True and hastension == False:
                 cmd1.append('--predict')
                 cmd1.append('tension')
            if (pdur == True or ppit == True or pbre == True or pten == True or prene == True) and pvoc == False and voc_status == True and hasvoicing == False:
                 cmd1.append('--predict')
                 cmd1.append('voicing')
            print(L('inf1'))
            subprocess.run(cmd1, check=True, cwd=diffolder, env=inference_env)
            cmd2 = [realpython, 'scripts/infer.py', 'acoustic', postvardsQ, '--exp', acockptQ, '--spk', spkname, '--lang', langname, '--out', dslocQ, '--title', dsname2Q]
            subprocess.run(cmd2, check=True, cwd=diffolder, env=inference_env)
            print(L('inf2'))

            play_audio(renderedQ)
      else:
            cmd3 = [realpython, 'scripts/infer.py', 'acoustic', dsinputQ, '--exp', acockptQ, '--spk', spkname, '--lang', langname, '--out', dslocQ, '--title', dsname2Q]
            print(L('inf2'))
            subprocess.run(cmd3, check=True, cwd=diffolder, env=inference_env)
            play_audio(renderedQ)

def replay():
    try:
        play_audio(renderedQ)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
         print(f"{L('replayerror')}: {error}")




app = ctk.CTk()
app.title(L('quick_inference_title'))
app_icon = ImageTk.PhotoImage(file=os.path.join(str(ASSETS_ROOT), "difftrainer-icon.png"))
app.iconphoto(False, app_icon)

lang = ctk.StringVar(value=guisettings['lang'])
L = ezlocalizr(language=lang.get(),
                            string_path=str(STRINGS_ROOT),
                            default_lang='en_US')
if lang.get() in ['jp-JP']:
      font = ctk.CTkFont(family=font_jp, size = 14)
elif lang.get() in ['zh-CN']:
      font = ctk.CTkFont(family=font_cn, size = 16)
elif lang.get() in ['zh-TW']:
      font = ctk.CTkFont(family=font_tw, size = 16)
else:
      font = ctk.CTkFont(family=font_en)

button = ctk.CTkButton(app, text=L('varckpt'), command=var, font=font)
button.grid(row=0, column=0, padx=10, pady=10)
button2 = ctk.CTkButton(app, text=L('acockpt'), command=aco, font=font)
button2.grid(row=0, column=1, padx=10, pady=10)

overwriteframe = ctk.CTkFrame(app)
overwriteframe.grid(row=1, padx=10, pady=10, column=0, columnspan=2)
over = ctk.CTkLabel(overwriteframe, text=L('overwrite'), font=font)
over.grid(row=0, column=0, columnspan=2, padx=5, pady=(5,0))
overdur = tk.BooleanVar()
overpit = tk.BooleanVar()
overbre = tk.BooleanVar()
overene = tk.BooleanVar()
overten = tk.BooleanVar()
overvoc = tk.BooleanVar()

durbox = ctk.CTkCheckBox(overwriteframe, text=L('duration'), variable=overdur, font=font)
durbox.grid(row=1, column=0, padx=(5,0), pady=5)
pitbox = ctk.CTkCheckBox(overwriteframe, text=L('pitch'), variable=overpit, font=font)
pitbox.grid(row=1, column=1, padx=(5,0), pady=5)
brebox = ctk.CTkCheckBox(overwriteframe, text=L('breathiness'), variable=overbre, font=font)
brebox.grid(row=2, column=0, padx=(5,0), pady=(0,5))
enebox = ctk.CTkCheckBox(overwriteframe, text=L('energy'), variable=overene, font=font)
enebox.grid(row=2, column=1, padx=(5,0), pady=(0,5))
tenbox = ctk.CTkCheckBox(overwriteframe, text=L('tension'), variable=overten, font=font)
tenbox.grid(row=3, column=0, padx=(5,0), pady=(0,5))
vocbox = ctk.CTkCheckBox(overwriteframe, text=L('voicing'), variable=overvoc, font=font)
vocbox.grid(row=3, column=1, padx=(5,0), pady=(0,5))



spk = tk.StringVar(value=L('spk'))
box = ctk.CTkEntry(app, textvariable=spk, font=font)
box.grid(row=2, column=0, padx=10, pady=10)
lcode = tk.StringVar(value=L('langcode'))
box2 = ctk.CTkEntry(app, textvariable=lcode, font=font)
box2.grid(row=2, column=1, padx=10, pady=10)
button3 = ctk.CTkButton(app, text=L('getds'), command=ds, font=font)
button3.grid(row=3, column=0, padx=10, pady=10)
button4 = ctk.CTkButton(app, text=L('render'), command=render, font=font)
button4.grid(row=3, column=1, padx=10, pady=10)
button5 = ctk.CTkButton(app, text=L('replay'), command=replay, font=font)
button5.grid(row=4, column=0, padx=10, pady=10, columnspan=2)

app.mainloop()
