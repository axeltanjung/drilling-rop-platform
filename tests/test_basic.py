from backend.utils.config import get_settings


def test_settings_defaults():
    settings = get_settings()
    assert settings.app_name == "StuckPipeAI"
    assert settings.backend_port == 8000
    assert settings.default_model == "xgboost"
    assert settings.prediction_threshold == 0.5


def test_risk_level_enum():
    from backend.api.schemas import RiskLevel
    assert RiskLevel.CRITICAL.value == "critical"
    assert RiskLevel.NORMAL.value == "normal"


def test_alert_severity_enum():
    from backend.api.schemas import AlertSeverity
    assert AlertSeverity.HIGH.value == "high"
    assert AlertSeverity.MEDIUM.value == "medium"
