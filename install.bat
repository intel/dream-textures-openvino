:<<BATCH
    @echo off
    echo **** OpenVino Dream Texture Addon Setup started **** 
    "C:\Program Files\Blender Foundation\Blender 3.4\3.4\python\bin\python.exe" -m ensurepip
    "C:\Program Files\Blender Foundation\Blender 3.4\3.4\python\bin\python.exe" -m pip install Pillow huggingface_hub
    "C:\Program Files\Blender Foundation\Blender 3.4\3.4\python\bin\python.exe" -m pip install -r requirements/win-openvino.txt --target .python_dependencies --upgrade to force replacement
    echo **** OpenVino Dream Texture Addon Setup Ended ****

    echo -----------------------------------------------------------------------------------------------
	echo -----------------------------------------------------------------------------------------------


	set /p model_setup= "Do you want to continue setting up the models for all the blender add-on now? Enter Y/N:  "

    
    echo your choice %model_setup%
	if %model_setup%==Y (
		set "continue=y"
	) else if %model_setup%==y (
		set "continue=y"
	) else ( set "continue=n"
		)



	if %continue%==y (
		echo **** OpenVINO MODELS SETUP STARTED ****
		"C:\Program Files\Blender Foundation\Blender 3.4\3.4\python\bin\python.exe" model_download.py

		echo **** OPENVINO MODELS SETUP COMPLETE **** 
		) else ( echo Model setup skipped. Please make sure you have all the required models setup.
		)
		
		
    exit /b 

    BATCH       