from pydantic import BaseModel, ConfigDict, Field

class NPVRequest(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )

    feature_1: str = Field(alias="Feature 1")
    feature_2: str = Field(alias="Feature 2")
    feature_3: str = Field(alias="Feature 3")
    feature_4: str = Field(alias="Feature 4")
    feature_5: str = Field(alias="Feature 5")
    feature_6: str = Field(alias="Feature 6")
    feature_7: str = Field(alias="Feature 7")
    feature_8: str = Field(alias="Feature 8")
    feature_9: str = Field(alias="Feature 9")
    feature_10: str = Field(alias="Feature 10")


class NPVResponse(BaseModel):

    predicted_npv: float
    model: str
