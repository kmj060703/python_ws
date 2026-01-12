# src/utils/config.py
import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Ensure required sections exist
    if 'paths' not in config:
        raise ValueError("Missing 'paths' section in config")
    if 'keys' not in config:
        raise ValueError("Missing 'keys' section in config")
    if 'scenario' not in config:
        raise ValueError("Missing 'scenario' section in config")
    if 'index_weights' not in config:
        config['index_weights'] = {
            'need': {
                'suicide_rate': 1,
                'depression_experience_rate': 1,
                'perceived_stress_rate': 1,
                'high_risk_drinking_rate': 1,
                'unmet_medical_need_rate': 1,
                'unemployment_rate': 1,
                'elderly_population_rate': 1,
                'old_dependency_ratio': 1,
                'single_households': 1,
                'basic_livelihood_recipients': 1
            },
            'supply': {
                'welfare_budget_per_capita': 1,
                'public_sports_facilities_count': 1,
                'parks_count': 1,
                'libraries_count': 1,
                'medical_institutions_count': 1,
                'health_promotion_centers_count': 1,
                'elderly_leisure_welfare_facilities_count': 1,
                'in_home_elderly_welfare_facilities_count': 1,
                'cultural_satisfaction': 1
            }
        }
    
    if 'modeling' not in config:
        config['modeling'] = {
            'target_col': 'suicide_rate',
            'random_state': 42
        }
    
    if 'scenario' not in config:
        config['scenario'] = {
            'parks_pct': 0.10,
            'health_center_add': 1,
            'medical_pct': 0.10
        }
    
    return config
