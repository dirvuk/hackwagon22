import pandas as pd
import lightgbm as lgb

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Body

model = lgb.Booster(model_file='model.dump')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

codes = pd.read_csv('codes.csv')


def get_name_by_code(code):
    try:
        return codes[codes['code'] == code]['name'].values[0]
    except:
        return 'Unknown'


@app.post("/predict/")
async def make_predict(payload: dict = Body(...)):
    st_code_snd = payload['st_code_snd'] or None
    st_code_rsv = payload['st_code_rsv'] or None
    date_depart_year = payload['date_depart_year'] if 'date_depart_year' in payload else None
    date_depart_month = payload['date_depart_month'] if 'date_depart_month' in payload else None
    date_depart_day = payload['date_depart_day'] if 'date_depart_day' in payload else None
    date_depart_week = payload['date_depart_week'] if 'date_depart_week' in payload else 30
    date_depart_hour = payload['date_depart_hour'] if 'date_depart_hour' in payload else None
    fr_id = payload['fr_id'] if 'fr_id' in payload else None
    route_type = payload['route_type'] if 'route_type' in payload else None
    is_load = payload['is_load'] if 'is_load' in payload else None
    rod = payload['rod'] if 'rod' in payload else None
    common_ch = payload['common_ch'] if 'common_ch' in payload else None
    vidsobst = payload['vidsobst'] if 'vidsobst' in payload else None
    distance = payload['distance'] if 'distance' in payload else None
    snd_org_id = payload['snd_org_id'] if 'snd_org_id' in payload else None
    rsv_org_id = payload['rsv_org_id'] if 'rsv_org_id' in payload else None
    snd_roadid = payload['snd_roadid'] if 'snd_roadid' in payload else None
    rsv_roadid = payload['rsv_roadid'] if 'rsv_roadid' in payload else None
    snd_dp_id = payload['snd_dp_id'] if 'snd_dp_id' in payload else None
    rsv_dp_id = payload['rsv_dp_id'] if 'rsv_dp_id' in payload else None
    avg_speed = payload['avg_speed'] if 'avg_speed' in payload else 16.2

    test_dict = {'st_code_snd': st_code_snd,
                 'st_code_rsv': st_code_rsv,
                 'date_depart_year': date_depart_year,
                 'date_depart_month': date_depart_month,
                 'date_depart_week': date_depart_week,
                 'date_depart_day': date_depart_day,
                 'date_depart_hour': date_depart_hour,
                 'fr_id': fr_id,
                 'route_type': route_type,
                 'is_load': is_load,
                 'rod': rod,
                 'common_ch': common_ch,
                 'vidsobst': vidsobst,
                 'distance': distance,
                 'snd_org_id': snd_org_id,
                 'rsv_org_id': rsv_org_id,
                 'snd_roadid': snd_roadid,
                 'rsv_roadid': rsv_roadid,
                 'snd_dp_id': snd_dp_id,
                 'rsv_dp_id': rsv_dp_id,
                 'avg_speed': avg_speed
                 }

    df = pd.DataFrame().from_dict(test_dict, orient='index').T
    predict = model.predict(df)

    avg_speed = 16.02
    if distance is not None:
        avg_speed = distance / predict[0]

    source_name = get_name_by_code(st_code_snd)
    destination_name = get_name_by_code(st_code_rsv)

    return {"time": predict[0],
            "st_code_snd": source_name,
            "st_code_rsv": destination_name,
            "avg_speed": avg_speed
            }
