"""Physics modules for soil moisture prediction."""
from smps.physics.water_balance import (
    TwoBucketWaterBalance,
    SiteSpecificWaterBalance,
    ModelParameters,
    BucketState,
    Fluxes,
    # Three-layer model for 0-100cm depth (FLDAS compatible)
    ThreeLayerWaterBalance,
    ThreeLayerParameters,
    create_three_layer_model,
)
from smps.physics.pedotransfer import (
    TextureClass,
    classify_soil_texture,
)

__all__ = [
    "TwoBucketWaterBalance",
    "SiteSpecificWaterBalance",
    "ModelParameters",
    "BucketState",
    "Fluxes",
    # Three-layer model exports
    "ThreeLayerWaterBalance",
    "ThreeLayerParameters",
    "create_three_layer_model",
    # Pedotransfer
    "TextureClass",
    "classify_soil_texture",
]
