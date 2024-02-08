import logging
from datetime import datetime

from flask import request
from flask_restful import Resource
from pydantic.error_wrappers import ValidationError

from settings import Settings
from v1.routines.plant_segmentation import PlantSegmentation
from v1.schemas.payloads import (
    SegmentPayload,
    ModelResponse,
    ErrorDescription,
    Response,
)


class ApiSegment(Resource):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cfg = Settings()
        self.plant_segmentation = PlantSegmentation(self.cfg, self.cfg.model)

    def get(self):
        return {"success": "ok"}, 200

    def post(self):
        json_data = request.get_json(force=True)

        self.logger.info(f"Processing started at {str(datetime.now())}")
        try:
            SegmentPayload.parse_obj(json_data)
            pred_label, data = self.plant_segmentation.main_routine(json_data)

        except (ValidationError, Exception) as e:
            error_description = ErrorDescription(
                raised=type(e).__name__,
                raisedOn="ApiSegment",
                message=str(e),
                code="400",
            )
            response = Response(
                message="failed",
                data=None,
                error=error_description,
                version=self.cfg.version,
            )
            return response.dict(), 400

        response = Response(
            message="success",
            data=ModelResponse(
                pred_label=str(pred_label),
                data=data,
            ),
            error=None,
            version=self.cfg.version,
        )
        self.logger.info(f"Request processed with success at {str(datetime.now())}")
        return response.dict(), 200
