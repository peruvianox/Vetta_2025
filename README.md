Quick description of what this is

#### Setup

This project uses uv to manage dependencies. Follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

Next setup a virtual environment and install dependencies:

bash```
uv venv
uv pip install -e .
```

Now let's load some subject data.

Create a folder called 'subjects' with these files nested underneath:

```
subjects/johndoe/info.json
subjects/johndoe/trials/trial1/raw-sensors.txt
subjects/johndoe/trials/trial1/raw-treadmill.mot
```

The `example-subjects/` folder contains an example (Note: the raw data files in the examples folder are empty. They only exist to demo the project structure and won't work with process-trial.ipynb)

That's it! You are ready to process a trial.

#### Project Structure

The project has three main parts:

subjects/               - The top-level directory containing subject data
process-trial.ipynb     - A notebook for preparing newly collected data from a trial
test-model.ipynb        - A notebook for testing processed data with the MLP model
utils/                  - Helper utilities used in notebooks
tf_model_5_RP.onnx      - The MLP model

The project assumes subjects is organized as follows:

```
subjects/
        ├── johndoe/
        │   ├── info.json
        │   └── trials/
        │       ├── trial-1/
        │       │   ├── raw-sensors.txt
        │       │   ├── raw-treadmill.mot
        │       │   └── stances/   <--- Produced by process-trial.ipynb
        │       ├── trial-2/
        │       └── trial-3/
        ├── janedoe/
        │   └── ...
        └── jimmydoe/
            └── ...
```

#### Usage

When a new trial is collected, save the info.json, raw-sensors.txt, and raw-treadmill.mot according to
the assumed project structure defined above.

Then follow the instructions in process-trial.ipynb to create stances from the raw data. The results
will appear in a directory called subjects/SUBJECT_NAME/trials/TRIAL_NAME/stances/

#### Procedure For Processing New Trial Data

##### Parse and filter acceleration data from raw-sensors.txt (5 stages)
1. Divide the raw data into 3 signals (waist, left ankle, right ankle)
2. Normalize acceleration vectors into scaler magnitudes
3. Resample each signal to 100 Hz
4. Divide each signal by gravity, converting to Gs
5. Copy and filter each signal with a butterworth filter applied forward and backwards over the original signal

##### Parse and filter vertical ground force (vGRF) data from raw-treadmill.mot (4 stages)
1. Divide the raw data into 2 signals (left and right)
2. Normalize vGRF vectors into scaler magnitudes
3. Resample each signal to 100 Hz
4. Divide each signal by subject weight, converting to ???

##### Align signals
Sensor signals have synchronized, absolute time stamps. Treadmill signals do not (time starts at zero).
process-trial.ipynb includes a little web app that allows the user to visually align the signals such that they
both have absolute timestamps (TODO: This is only 99% true).

##### Create stances
1. Use scipy.signal.find_peaks() to find peaks and peak widths in the left and right vGRF signals
2. Split each signal into stances, whos indices are defined by the left boundary of each vGRF peak
3. Resample each stance to 100 frames and save

#### TODO

[] Validate the stomp alignment UI
[] Verify the accel and vGRF normalization procedure
[] Fix the tiny timestamp alignment issue
[] Verify the procedure for retrieving stances
[] Stress test the project and load in a bunch of data and test with the MLP model
