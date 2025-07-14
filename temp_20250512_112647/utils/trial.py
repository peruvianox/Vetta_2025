from typing import List
import pandas as pd
import os
import json

from utils.sensor import load_sensors_data
from utils.treadmill import load_treadmill_data

# Constants
SENSORS_FILENAME = 'raw-sensors.txt'
TREADMILL_FILENAME = 'raw-treadmill.mot'

class Trial:
    """Container for all data from a single trial.
    
    This class manages the loading and processing of sensor and treadmill data
    for a single trial, including stance phase detection and storage.
    
    Attributes:
        path: Path to the trial directory
        name: Trial identifier
        subject_name: Name of the subject
        subject_weight: Weight of the subject in kg
    """
    def __init__(self, path: str, subject_name: str, subject_weight: float) -> None:
        """Initialize trial with path and subject information.
        
        Args:
            path: Path to trial directory
            subject_name: Name of the subject
            subject_weight: Weight of the subject in kg
        """
        self.path = path
        self.name = os.path.basename(path)
        self.subject_name = subject_name
        self.subject_weight = subject_weight

    def load_sensors(self) -> pd.DataFrame:
        """Load and process sensor data from the trial directory.
        
        Returns:
            DataFrame containing processed sensor data
        """
        return load_sensors_data(os.path.join(self.path, SENSORS_FILENAME))

    def load_treadmill(self) -> pd.DataFrame:
        """Load and process treadmill data from the trial directory.
        
        Returns:
            DataFrame containing processed treadmill data
        """
        return load_treadmill_data(os.path.join(self.path, TREADMILL_FILENAME), self.subject_weight)

def load_trials(subject_path: str) -> List[Trial]:
    """Load all trials for a subject.
    
    Args:
        subject_path: Path to the subject directory
        
    Returns:
        List of trial data objects
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If subject info is invalid
    """
    # Get subject name from path
    subject_name = os.path.basename(subject_path)

    # Load subject info
    info_path = os.path.join(subject_path, 'info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"info.json is missing from {subject_path}")

    with open(info_path, 'r', encoding='utf-8') as fp:
        info = json.load(fp)
        if 'weight' not in info:
            raise ValueError("info.json is missing the weight field")
        subject_weight = float(info['weight'])

    # Load trials
    trials_dir = os.path.join(subject_path, 'trials')
    if not os.path.exists(trials_dir):
        raise FileNotFoundError(f"trials directory is missing from {subject_path}")

    results = []
    for trial_name in os.listdir(trials_dir):
        trial_dir = os.path.join(trials_dir, trial_name)
        if trial_name.startswith('.') or not os.path.isdir(trial_dir):
            continue
        contents = list(os.listdir(trial_dir))

        # Check for required files
        if SENSORS_FILENAME not in contents:
            raise FileNotFoundError(f"{SENSORS_FILENAME} missing from {trial_dir}")
        if TREADMILL_FILENAME not in contents:
            raise FileNotFoundError(f"{TREADMILL_FILENAME} missing from {trial_dir}")

        # Create trial object
        trial = Trial(trial_dir, subject_name, subject_weight)
        results.append(trial)

    return results

def load_trial(subjects_path: str, subject_name: str, trial_name: str) -> Trial:
    """Load a single trial for a subject.
    
    Args:
        subjects_path: Path to subjects directory
        subject_name: Name of the subject
        trial_name: Name of the trial
        
    Returns:
        Trial object containing the trial data
    """
    trial_path = os.path.join(subjects_path, subject_name, 'trials', trial_name)
    if not os.path.exists(trial_path):
        raise ValueError(f"Trial path does not exist: {trial_path}")

    # Load subject info
    subject_info_path = os.path.join(subjects_path, subject_name, 'info.json')
    with open(subject_info_path, 'r', encoding='utf-8') as f:
        subject_info = json.load(f)

    return Trial(trial_path, subject_name, subject_info['weight'])
