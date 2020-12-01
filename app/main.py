from typing import Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from svm.water_classification import train_drip_model, train_on_model
from utilities.enums import WaterState
from utilities.annotations import Seconds, Recording
from utilities.constants import version

from data_recording.record_sample import record_sample_for_
from svm.water_classification import filename

from feature_extraction.features import generate_features_for_

app = FastAPI(title="Water Status Audio Classification",
              description="A project that uses a hand crafted SVM to classify audio coming from a sink",
              version=version
              )


class WaterManager:
    class __WaterManager:
        def __init__(self):
            pass

    _instance = None
    status: WaterState = None

    df = pd.read_csv(f"svm/{filename}")
    drip_model = train_drip_model(df)
    on_model = train_on_model(df)

    def __init__(self):
        self.record_length: Seconds = 4

        if not WaterManager._instance:
            WaterManager._instance = WaterManager.__WaterManager()

    def get_status(self) -> Optional[WaterState]:
        recording: Recording = record_sample_for_(self.record_length)
        amplitude, peak_count = generate_features_for_(recording)

        is_dripping = self.drip_model.predict(amplitude, peak_count)
        is_on = self.on_model.predict(amplitude, peak_count)

        if np.sign(is_dripping) == -1:
            self.status = WaterState.DRIP
        elif np.sign(is_on) == -1:
            self.status = WaterState.ON
        else:
            self.status = WaterState.OFF

        return self.status

    def __getattr__(self, name):
        return getattr(self.instance, name)


class WaterStateSchema(BaseModel):
    water_status: str

    class Config:
        schema_extra = {
            "water_status": "OFF|DRIP|ON"
        }


@app.get("/")
async def root():
    return {
        "welcome_message": "I'd reccomend you go to this link: http://127.0.0.1:8000/docs#/default/get_water_status_status_get"}


@app.get("/status", response_model=WaterStateSchema,
         status_code=200,
         summary="Get's the latest water drip status",
         description="If you're hosting this on your computer, this will record 4 seconds of audio from your computer's"
                     " microphone and will then tell you if a drip was detected."
                     "\n\nOptions include (strings): ( OFF | DRIP | ON | UNKNOWN ) ")
async def get_water_status():

    manager = WaterManager()
    water_status = manager.get_status()

    if water_status is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Unable to classify")

    return WaterStateSchema(
        water_status=water_status.name
    )
