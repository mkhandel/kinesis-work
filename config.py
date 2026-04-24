"""
Mehek's work:
Configuration and rule definitions for the ETL pipeline.

This module contains modality-agnostic configuration that allows
new data sources to be added by defining rules, not code changes.

NOTE: Rule values (event types, measure rules, source weights, normalization)
are starter defaults for testing. Domain teams (ASU, Feb, ML) refine and expand
them—see docs/project/OWNERSHIP.md and docs/reference/rule_formats.md.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    url: str
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load database config from environment variables."""
        db_url = os.getenv("SUPABASE_DB_URL", "").strip()  # Strip whitespace
        if db_url:
            # Parse PostgreSQL URL
            # Format: postgresql://user:password@host:port/database
            # Also supports pooler format: postgresql://user.project:password@host:port/database
            import re
            match = re.match(
                r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url
            )
            if match:
                user, password, host, port, database = match.groups()
                return cls(
                    url=db_url,
                    host=host,
                    port=int(port),
                    database=database,
                    user=user,
                    password=password,
                )
        
        # Fallback to individual env vars
        return cls(
            url=os.getenv("SUPABASE_DB_URL", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    batch_size: int = 1000
    log_level: str = "INFO"
    environment: str = "development"
    default_timezone: str = "UTC"
    geo_bucket_precision: int = 4
    require_consent: bool = True
    default_consent_scope: List[str] = None

    def __post_init__(self):
        if self.default_consent_scope is None:
            self.default_consent_scope = ["analytics"]

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load pipeline config from environment variables."""
        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", "1000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            environment=os.getenv("ENVIRONMENT", "development"),
            default_timezone=os.getenv("DEFAULT_TIMEZONE", "UTC"),
            geo_bucket_precision=int(os.getenv("GEO_BUCKET_PRECISION", "4")),
            require_consent=os.getenv("REQUIRE_CONSENT", "true").lower() == "true",
            default_consent_scope=os.getenv("DEFAULT_CONSENT_SCOPE", "analytics").split(","),
        )


# Event type mapping rules
# Modality-agnostic: new sources just add new rules here
EVENT_TYPE_RULES: Dict[str, Dict[str, str]] = {
    "wifi": {
        "wifi_probe": "presence",
        "wifi_connection": "presence",
    },
    "qr": {
        "qr_scan": "engagement",
        "qr_tap": "engagement",
    },
    "text": {
        "sms_feedback": "feedback",
    },
    "survey": {
        "micro_survey": "feedback",
        "survey_response": "feedback",
        "survey_completion": "feedback",
    },
    "mobile": {
        "mobile_interaction": "engagement",
        "button_click": "engagement",
    },
    # toronto_311: Minimal starter for testing; ASU/Feb refine and expand.
    "toronto_311": {
        # Base fallback
        "service_request": "feedback",

        # Category-specific modalities (derived upstream from Service Request Type)
        "service_request_noise": "feedback",
        "service_request_waste": "feedback",
        "service_request_roads": "feedback",
        "service_request_water_sewer": "feedback",
        "service_request_property": "feedback",
        "service_request_trees": "feedback",
        "service_request_animal": "feedback",
        "service_request_other": "feedback",
    },
}


# Geo bucket derivation rules
def derive_geo_bucket(site_id: str, zone: Optional[str] = None) -> str:
    """
    Derive geographic bucket from site_id and zone.
    
    This is a simple implementation. In production, this might:
    - Look up coordinates from a site registry
    - Use geohashing
    - Apply custom geographic boundaries
    """
    if zone:
        return f"{site_id}:{zone}"
    return site_id


# Entity scope derivation rules
def derive_entity_scope(site_id: str, zone: Optional[str] = None) -> str:
    """
    Derive entity scope from site_id and zone.
    
    Returns: 'site', 'zone', 'city', 'region', etc.
    """
    if zone:
        return "zone"
    return "site"


# Source weight rules
# Weight per source for aggregation: higher = more trusted/important when combining data.
# Used when aggregating normalized events across sources (e.g. survey=1.5, 311=0.9).
# Domain teams (ASU, Feb) refine these—values are starter defaults for testing.
SOURCE_WEIGHTS: Dict[str, float] = {
    "wifi": 1.0,
    "qr": 1.2,  # QR scans indicate explicit engagement
    "text": 1.0,  # Text feedback is valuable but may be less structured
    "survey": 1.5,  # Survey responses are high-value
    "mobile": 1.0,
    # toronto_311: Minimal starter for testing; ASU/Feb refine and expand.
    "toronto_311": 1.5, #high-intent engagement because someone filed a complaint or request to the city
}
# Measure extraction rules
# Defines how to extract numeric values from different modalities
MEASURE_EXTRACTION_RULES: Dict[str, Dict[str, Any]] = {
    "wifi": {
        "device_count": {
            "path": "payload.device_count",
            "unit": "count",
            "confidence": 0.9,
        },
        "signal_strength": {
            "path": "payload.signal_strength",
            "unit": "dBm",
            "confidence": 0.8,
        },
        "dwell_seconds": {
            "path": "payload.dwell_seconds",
            "unit": "seconds",
            "confidence": 0.8,
        },
    },
    "qr": {
        "tap": {
            "path": "outcome.tap",  # Try outcome field first
            "fallback_path": "payload.action",  # Fallback to payload
            "value_map": {"true": 1, "false": 0, True: 1, False: 0, "tap": 1, "scan": 1, "none": 0},
            "unit": "count",
            "confidence": 1.0,
        },
        "opt_in": {
            "path": "outcome.opt_in",  # Try outcome field first
            "fallback_path": "payload.opt_in",  # Fallback to payload
            "value_map": {"true": 1, "false": 0, True: 1, False: 0},
            "unit": "boolean",
            "confidence": 1.0,
        },
    },
    "text": {
        "sentiment": {
            "path": "text_input",
            "extractor": "sentiment_analyzer",  # Special handler
            "unit": "score",
            "confidence": 0.7,
        },
    },
    "survey": {
        "response_score": {
            "path": "payload.response_score",
            "unit": "score",
            "confidence": 1.0,
        },
        "sentiment": {
            "path": "text_input",
            "extractor": "sentiment_analyzer",  # Special handler (if text_input exists)
            "unit": "score",
            "confidence": 0.7,
        },
    },
    "mobile": {
        "interaction_count": {
            "path": "payload.interaction_type",
            "value_map": {"button_click": 1, "swipe": 1, "none": 0},
            "unit": "count",
            "confidence": 0.9,
        },
        "duration": {
            "path": "payload.duration_ms",
            "unit": "milliseconds",
            "confidence": 0.8,
        },
    },
    # toronto_311: Minimal starter for testing; ASU/Feb refine and expand.
    "toronto_311": {
        # Always 1 row = 1 request
        "request_count": {
            "value": 1,
            "unit": "count",
            "confidence": 1.0,
        },

        # One-hot-ish status indicators (lets you aggregate rates later)
        "is_completed": {
            "path": "payload.status",
            "value_map": {"Completed": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_cancelled": {
            "path": "payload.status",
            "value_map": {"Cancelled": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_in_progress": {
            "path": "payload.status",
            "value_map": {"In Progress": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_new": {
            "path": "payload.status",
            "value_map": {"New": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_closed": {
            "path": "payload.status",
            "value_map": {"Closed": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },

        # Compact “outcome” score for quick dashboards:
        # +1 = resolved (Completed/Closed), -1 = cancelled, 0 = still open/unknown
        "status_outcome_score": {
            "path": "payload.status",
            "value_map": {
                "Completed": 1,
                "Closed": 1,
                "Cancelled": -1,
                "In Progress": 0,
                "New": 0,
                "Unknown": 0,
            },
            "default": 0,
            "unit": "status_outcome",
            "confidence": 1.0,
        },

        #for division types:
        "is_311_division": {
            "path": "payload.division",
            "value_map": {"311": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_licensing_and_standards_division": {
            "path": "payload.division",
            "value_map": {"Municipal Licensing & Standards": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_parks_and_rec_division": {
            "path": "payload.division",
            "value_map": {"Parks and Recreation": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_waste_management_division": {
            "path": "payload.division",
            "value_map": {"Solid Waste Management Services": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_water_division": {
            "path": "payload.division",
            "value_map": {"Toronto Water": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_transportation_division": {
            "path": "payload.division",
            "value_map": {"Transportation Services": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },

        #compact division rule
        "division_score": {
            "path": "payload.division",
            "value_map": {
                "311": 1,
                "Municipal Licensing & Standards": 2,
                "Parks and Recreation": 3,
                "Solid Waste Management Services": 4,
                "Toronto Water": 5,
                "Transportation Services": 6,
            },
            "default": 0,
            "unit": "division_outcome",
            "confidence": 1.0,
        },

        #for section types:
        "is_business_licensing_enforcement_section": {
            "path": "payload.section",
            "value_map": {"Business Licensing Enforcement": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_business_operations_management_section": {
            "path": "payload.section",
            "value_map": {"Business Operations Management": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_bylaw_enforcement_section": {
            "path": "payload.section",
            "value_map": {"Bylaw Enforcement": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_collections_section": {
            "path": "payload.section",
            "value_map": {"Collections": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_collections_and_litter_operations_section": {
            "path": "payload.section",
            "value_map": {"Collections and Litter Operations": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_district_ops_section": {
            "path": "payload.section",
            "value_map": {"District Ops": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_investigation_services_section": {
            "path": "payload.section",
            "value_map": {"Investigation Services": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_litter_operations_section": {
            "path": "payload.section",
            "value_map": {"Litter Operations": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_operations_section": {
            "path": "payload.section",
            "value_map": {"Operations": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_parks_section": {
            "path": "payload.section",
            "value_map": {"Parks": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_parks_enforcement_section": {
            "path": "payload.section",
            "value_map": {"Parks Enforcement": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_permits_and_enforcement_section": {
            "path": "payload.section",
            "value_map": {"Permits & Enforcement": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_right_of_way_section": {
            "path": "payload.section",
            "value_map": {"Right of Way (ROW)": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_road_operations_section": {
            "path": "payload.section",
            "value_map": {"Road Operations": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_tmc_section": {
            "path": "payload.section",
            "value_map": {"TMC": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_toronto_animal_services_section": {
            "path": "payload.section",
            "value_map": {"Toronto Animal Services": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_traffic_management_section": {
            "path": "payload.section",
            "value_map": {"Traffic Management": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_traffic_ops_section": {
            "path": "payload.section",
            "value_map": {"Traffic Ops": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_traffic_safety_section": {
            "path": "payload.section",
            "value_map": {"Traffic Safety": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_waste_enforcement_section": {
            "path": "payload.section",
            "value_map": {"Waste Enforcement": 1},
            "default": 0,
            "unit": "binary_01",
            "confidence": 1.0,
        },

        #compact division rule
        "section_score": {
            "path": "payload.section",
            "value_map": {
                "Business Licensing Enforcement": 1,
                "Business Operations Management": 2,
                "Bylaw Enforcement": 3,
                "Collections": 4,
                "Collections and Litter Operations": 5,
                "District Ops": 6,
                "Investigation Services": 7,
                "Litter Operations": 8,
                "Operations": 9,
                "Parks": 10,
                "Parks Enforcement": 11,
                "Permits & Enforcement": 12,
                "Right of Way (ROW)": 13,
                "Road Operations": 14,
                "TMC": 15,
                "Toronto Animal Services": 16,
                "Traffic Management": 17,
                "Traffic Ops": 18,
                "Traffic Safety": 19,
                "Waste Enforcement": 20
            },
            "default": 0,
            "unit": "section_outcome",
            "confidence": 1.0,
        },

        # Optional: if your ingestion step adds these fields (recommended)
        "is_weekend": {
            "path": "payload.is_weekend",
            "value_map": {True: 1, False: 0, "true": 1, "false": 0},
            "unit": "binary_01",
            "confidence": 1.0,
        },
        "is_night": {
            "path": "payload.is_night",
            "value_map": {True: 1, False: 0, "true": 1, "false": 0},
            "unit": "binary_01",
            "confidence": 1.0,
        },
    },
}


# Normalization rules
# Defines how to normalize different measure types to -1 to +1 scale
NORMALIZATION_RULES: Dict[str, Dict[str, Any]] = {
    "likert_5": {
        "type": "linear",
        "min": 1,
        "max": 5,
        "output_min": -1,
        "output_max": 1,
    },
    "likert_7": {
        "type": "linear",
        "min": 1,
        "max": 7,
        "output_min": -1,
        "output_max": 1,
    },
    "count": {
        "type": "log_scale",  # per_capita needs population; log_scale works for counts
    },
    "percentage": {
        "type": "linear",
        "min": 0,
        "max": 100,
        "output_min": -1,
        "output_max": 1,
    },
    "score": {
        "type": "linear",
        "min": 0,
        "max": 10,
        "output_min": -1,
        "output_max": 1,
    },
    # Sentiment from ML API is already in [-1,+1]; passthrough (no re-normalization)
    "sentiment_score": {
        "type": "linear",
        "min": -1,
        "max": 1,
        "output_min": -1,
        "output_max": 1,
    },

    #added
    "binary_01": {
        "type": "linear", #linear transformation is good for many measures that center their data around 0 (with -1 and 1 being the extremes/ends of the data)
        "min": 0,
        "max": 1,
        "output_min": -1,
        "output_max": 1,
    },
    "status_outcome": {
        "type": "linear",
        "min": -1,
        "max": 1,
        "output_min": -1,
        "output_max": 1,
    },
    "division_outcome": {
        "type": "linear",
        "min": 0,
        "max": 6,
        "output_min": 0,
        "output_max": 6,
    },
    "section_outcome": {
        "type": "linear",
        "min": 0,
        "max": 20,
        "output_min": 0,
        "output_max": 20,
    },
}


def get_event_type(source: str, modality: str) -> str:
    """
    Get civic event type from source and modality.
    
    This is where modality-agnostic design shines:
    new sources just need new rules, not code changes.
    """
    source_rules = EVENT_TYPE_RULES.get(source, {})
    return source_rules.get(modality, "unknown")


def get_source_weight(source: str) -> float:
    """Get weight for a data source."""
    return SOURCE_WEIGHTS.get(source, 1.0)


def get_measure_rules(source: str) -> Dict[str, Any]:
    """Get measure extraction rules for a source."""
    return MEASURE_EXTRACTION_RULES.get(source, {})


def get_normalization_rule(measure_type: str) -> Optional[Dict[str, Any]]:
    """Get normalization rule for a measure type."""
    return NORMALIZATION_RULES.get(measure_type)


# Database connection helper
def get_db_connection():
    """Get database connection using configuration."""
    import psycopg2
    config = DatabaseConfig.from_env()
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
    )
