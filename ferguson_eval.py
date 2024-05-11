import pandas as pd
import numpy as np
from difflib import SequenceMatcher

VARIERTY_PARAMETERS = {
    'Barbera': {
        'Hc_initial': -10.1,
        'Hc_min': -1.2,
        'Hc_max': -23.5,
        'T_threshold_endo': 15.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.06,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.10,
        'Deacclimation_rate_eco': 0.08,
        'Theta': 7
    },
    'Cabernet franc': {
        'Hc_initial': -9.9,
        'Hc_min': -1.2,
        'Hc_max': -25.4,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 4.0,
        'Ecodormancy_boundary': -500,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.04,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 7
    },
    'Cabernet Sauvignon': {
        'Hc_initial': -10.3,
        'Hc_min': -1.2,
        'Hc_max': -25.1,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.08,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 7
    },
    'Chardonnay': {
        'Hc_initial': -11.8,
        'Hc_min': -1.2,
        'Hc_max': -25.7,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -600,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.10,
        'Deacclimation_rate_eco': 0.08,
        'Theta': 7
    },
    'Chenin blanc': {
        'Hc_initial': -12.1,
        'Hc_min': -1.2,
        'Hc_max': -24.1,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 4.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.04,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 7
    },
    'Concord': {
        'Hc_initial': -12.8,
        'Hc_min': -2.5,
        'Hc_max': -29.5,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -600,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 3
    },
    'Dolcetto': {
        'Hc_initial': -10.1,
        'Hc_min': -1.2,
        'Hc_max': -23.2,
        'T_threshold_endo': 12.0,
        'T_threshold_eco': 4.0,
        'Ecodormancy_boundary': -600,
        'Acclimation_rate_endo': 0.16,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.10,
        'Deacclimation_rate_eco': 0.12,
        'Theta': 3
    },
    'Gewurztraminer': {
        'Hc_initial': -11.6,
        'Hc_min': -1.2,
        'Hc_max': -24.9,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 6.0,
        'Ecodormancy_boundary': -400,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.06,
        'Deacclimation_rate_eco': 0.18,
        'Theta': 5
    },
    'Grenache': {
        'Hc_initial': -10.0,
        'Hc_min': -1.2,
        'Hc_max': -22.7,
        'T_threshold_endo': 12.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -500,
        'Acclimation_rate_endo': 0.16,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.06,
        'Theta': 5
    },
    'Lemberger': {
        'Hc_initial': -13.0,
        'Hc_min': -1.2,
        'Hc_max': -25.6,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -800,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.18,
        'Theta': 7
    },
    'Malbec': {
        'Hc_initial': -11.5,
        'Hc_min': -1.2,
        'Hc_max': -25.1,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 4.0,
        'Ecodormancy_boundary': -400,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.08,
        'Deacclimation_rate_endo': 0.06,
        'Deacclimation_rate_eco': 0.08,
        'Theta': 7
    },
    'Merlot': {
        'Hc_initial': -10.3,
        'Hc_min': -1.2,
        'Hc_max': -25.0,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -500,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.04,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 7
     },
    'Mourvedre': {
        'Hc_initial': -9.5,
        'Hc_min': -1.2,
        'Hc_max': -22.1,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 6.0,
        'Ecodormancy_boundary': -600,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.06,
        'Deacclimation_rate_endo': 0.08,
        'Deacclimation_rate_eco': 0.14,
        'Theta': 5
    },
    'Nebbiolo': {
        'Hc_initial': -11.1,
        'Hc_min': -1.2,
        'Hc_max': -24.4,
        'T_threshold_endo': 11.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.16,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 3
    },
    'Pinot gris': {
        'Hc_initial': -12.0,
        'Hc_min': -1.2,
        'Hc_max': -24.1,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 6.0,
        'Ecodormancy_boundary': -400,
        'Acclimation_rate_endo': 0.12,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.20,
        'Theta': 3
    },
    'Riesling': {
        'Hc_initial': -12.6,
        'Hc_min': -1.2,
        'Hc_max': -26.1,
        'T_threshold_endo': 12.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.14,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.12,
        'Theta': 7
    },
    'Sangiovese': {
        'Hc_initial': -10.7,
        'Hc_min': -1.2,
        'Hc_max': -21.9,
        'T_threshold_endo': 11.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.14,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.06,
        'Theta': 7
    },
    'Sauvignon blanc': {
        'Hc_initial': -10.6,
        'Hc_min': -1.2,
        'Hc_max': -24.9,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -300,
        'Acclimation_rate_endo': 0.08,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.06,
        'Deacclimation_rate_eco': 0.12,
        'Theta': 7
    },
    'Semillon': {
        'Hc_initial': -10.4,
        'Hc_min': -1.2,
        'Hc_max': -22.4,
        'T_threshold_endo': 13.0,
        'T_threshold_eco': 7.0,
        'Ecodormancy_boundary': -300,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.02,
        'Deacclimation_rate_endo': 0.08,
        'Deacclimation_rate_eco': 0.20,
        'Theta': 5
    },
    'Sunbelt': {
        'Hc_initial': -11.8,
        'Hc_min': -2.5,
        'Hc_max': -29.1,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -400,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.06,
        'Deacclimation_rate_eco': 0.12,
        'Theta': 1.5
    },
    'Syrah': {
        'Hc_initial': -10.3,
        'Hc_min': -1.2,
        'Hc_max': -24.2,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 4.0,
        'Ecodormancy_boundary': -700,
        'Acclimation_rate_endo': 0.08,
        'Acclimation_rate_eco': 0.04,
        'Deacclimation_rate_endo': 0.06,
        'Deacclimation_rate_eco': 0.08,
        'Theta': 7
    },
    'Viognier': {
        'Hc_initial': -11.2,
        'Hc_min': -1.2,
        'Hc_max': -24.0,
        'T_threshold_endo': 14.0,
        'T_threshold_eco': 5.0,
        'Ecodormancy_boundary': -300,
        'Acclimation_rate_endo': 0.10,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.08,
        'Deacclimation_rate_eco': 0.10,
        'Theta': 7
    },
    'Zinfandel': {
        'Hc_initial': -10.4,
        'Hc_min': -1.2,
        'Hc_max': -24.4,
        'T_threshold_endo': 12.0,
        'T_threshold_eco': 3.0,
        'Ecodormancy_boundary': -500,
        'Acclimation_rate_endo': 0.16,
        'Acclimation_rate_eco': 0.10,
        'Deacclimation_rate_endo': 0.02,
        'Deacclimation_rate_eco': 0.06,
        'Theta': 7
    }
}

REGRESSION_COEFFICIENTS_LTE = {
    'Concord': {10: [0.6229, 0.9269], 90: [-2.0214, 0.9893]},
    'Chardonnay': {10: [1.3825, 1.0115], 90: [-2.1321, 0.9464], },
    'Cabernet Sauvignon': {10: [1.4694, 0.9995], 90: [-2.2381, 0.9476]},
    'Chenin Blanc': {10: [0.292662886, 0.87454158], 90: [-2.254780767, 0.980682525]},
    'Merlot': {10: [1.029118997, 0.977973987], 90: [-1.635189458, 0.981290014]},
    'Pinot Gris': {10: [1.582989951, 0.988794533], 90: [-2.218950917, 0.961894374]},
    'Sunbelt': {10: [0.821649143, 0.938924258], 90: [-1.654424259, 0.99910646]},
    'Syrah': {10: [0.909540563, 0.965709596], 90: [-1.565016554, 0.983274558]},
    'Viognier': {10: [1.25332805, 0.973212846], 90: [-1.623456923, 0.993659096]},
    'Riesling': {10: [1.048071008, 0.980701371], 90: [-1.696375957, 0.971074128]},
    'Barbera': {10: [1.226862729, 0.985455538], 90: [-1.206274797, 1.006025303]},
    'Cabernet Franc': {10: [1.154802908, 0.981339375], 90: [-2.307033559, 0.961380067]},
    'Dolcetto': {10: [0.724896941, 0.975466814], 90: [-1.24849354, 0.995781299]},
    'Grenache': {10: [0.110813195, 0.926237027], 90: [-0.321928235, 1.049048642]},
    'Gewurztraminer': {10: [2.167172843, 1.018047285], 90: [-1.662601904, 0.984359476]},
    'Lemberger': {10: [1.878306192, 0.997965079], 90: [-1.103743194, 1.012747248]},
    'Malbec': {10: [1.331545124, 0.982935173], 90: [-2.173687114, 0.966520326]},
    'Mourvedre': {10: [1.760280204, 1.000821375], 90: [-1.991327552, 0.969310935]},
    'Nebbiolo': {10: [1.13122912, 0.964593036], 90: [-1.666265924, 0.982690175]},
    'Sauvignon Blanc': {10: [0.764497777, 0.943945974], 90: [-1.516375842, 0.991982834]},
    'Sangiovese': {10: [0.803799223, 0.955647333], 90: [-0.992155767, 1.016722033]},
    'Semillon': {10: [0.62458141, 0.92571644], 90: [-1.075391026, 1.028846154]},
    'Zinfandel': {10: [0.899838214, 0.965112864], 90: [-0.973907989, 1.014978602]}
   }

GRAPE_VARIETIES = {'Aligote': 'AL', 'Alvarinho': 'AV', 'Auxerrois': 'AX', 'Barbera': 'BA', 'Cabernet Franc': 'CF',
                   'Cabernet Sauvignon': 'CS', 'Chardonnay': 'CH', 'Chenin Blanc': 'CB', 'Concord': 'CD',
                   'Dolcetto': 'DO', 'Durif': 'DR', 'Gewurztraminer': 'GW', 'Green Veltliner': 'GV', 'Grenache': 'GR',
                   'Lemberger': 'LM', 'Malbec': 'MB', 'Melon': 'ML', 'Merlot': 'MR', 'Mourvedre': 'MV',
                   'Muscat Blanc': 'MuB', 'Nebbiolo': 'NB', 'Petit Verdot': 'PV', 'Pinot Blanc': 'PB',
                   'Pinot Gris': 'PG', 'Pinot Noir': 'PN', 'Riesling': 'WR', 'Sangiovese': 'SG',
                   'Sauvignon Blanc': 'SB', 'Semillon': 'SM', 'Sunbelt': 'ST', 'Syrah': 'SY', 'Tempranillo': 'TM',
                   'Viognier': 'VG', 'Zinfandel': 'ZI'}

COLUMNS_RESULT = ['DATE', 'MEAN_AT', 'MIN_AT', 'MAX_AT', 'PREDICTED_LTE50', 'PREDICTED_BUDBREAK']


def _complete_lte_model_prediction(cultivar, predicted_lte50, option):
    return (REGRESSION_COEFFICIENTS_LTE[cultivar][option][0] +
            (REGRESSION_COEFFICIENTS_LTE[cultivar][option][1] * predicted_lte50))


def _cultivar_name_similarity(name, cultivar_names=list(GRAPE_VARIETIES.keys())):
    result = ""
    name = name.lower()
    ratios = [SequenceMatcher(None, name, cultivar.lower()).ratio() for cultivar in cultivar_names]
    if max(ratios) >= 0.6:
        found_name = cultivar_names[ratios.index(max(ratios))].lower()
        if name[:3] == found_name[0:3]:
            result = cultivar_names[ratios.index(max(ratios))]
    return result


def has_parameters_to_run_ferguson(cultivar):
    result = False
    cultivar = _cultivar_name_similarity(cultivar)
    if cultivar in VARIERTY_PARAMETERS.keys():
        result = True
    return cultivar, result


def ferguson_model(cultivar, df_input_temperatures):
    df_results = pd.DataFrame(columns=COLUMNS_RESULT)

    hc_initial = VARIERTY_PARAMETERS[cultivar]['Hc_initial']
    hc_min = VARIERTY_PARAMETERS[cultivar]['Hc_min']
    hc_max = VARIERTY_PARAMETERS[cultivar]['Hc_max']
    t_threshold = [VARIERTY_PARAMETERS[cultivar]['T_threshold_endo'],
                   VARIERTY_PARAMETERS[cultivar]['T_threshold_eco']]
    ecodormancy_boundary = VARIERTY_PARAMETERS[cultivar]['Ecodormancy_boundary']
    acclimation_rate = [VARIERTY_PARAMETERS[cultivar]['Acclimation_rate_endo'],
                       VARIERTY_PARAMETERS[cultivar]['Acclimation_rate_eco']]
    deacclimation_rate = [VARIERTY_PARAMETERS[cultivar]['Deacclimation_rate_endo'],
                          VARIERTY_PARAMETERS[cultivar]['Deacclimation_rate_eco']]
    theta = [1, VARIERTY_PARAMETERS[cultivar]['Theta']]

    #Calculate range of hardiness values possible, this is needed for the logistic component
    hc_range = hc_min - hc_max

    #Initialize variables for start of model
    dd_heating_sum = 0
    dd_chilling_sum = 0
    dormancy_period = 0
    base10_chilling_sum = 0
    model_hc_yesterday = hc_initial

    #Read data from "input_temps" for today
    for row in df_input_temperatures.itertuples():
        if row.MIN_AT == -100 or row.MIN_AT == -200:
            new_row = {'DATE': [row.DATE],
                       'MEAN_AT': [row.MEAN_AT],
                       'MIN_AT': [row.MIN_AT],
                       'MAX_AT': [row.MAX_AT],
                       'PREDICTED_LTE50': [np.nan],
                       'PREDICTED_BUDBREAK': [np.nan]}
            df_results = pd.concat([df_results, pd.DataFrame.from_dict(new_row)], ignore_index=True)
            continue

        #Calculate heating degree days for today used in deacclimation
        if row.MEAN_AT > t_threshold[dormancy_period]:
            dd_heating_today = row.MEAN_AT - t_threshold[dormancy_period]
        else:
            dd_heating_today = 0

        #Calculate cooling degree days for today used in acclimation
        if row.MEAN_AT <= t_threshold[dormancy_period]:
            dd_chilling_today = row.MEAN_AT - t_threshold[dormancy_period]
        else:
            dd_chilling_today = 0

        #Calculate cooling degree days using base of 10c to be used in dormancy release
        if row.MEAN_AT <= 10:
            base10_chilling_today = row.MEAN_AT - 10
        else:
            base10_chilling_today = 0

        #Calculate new model_hc for today
        deacclimation = dd_heating_today * deacclimation_rate[dormancy_period] * (
                1 - ((model_hc_yesterday - hc_max) / hc_range) ** theta[dormancy_period])

        #Do not allow deacclimation unless some chilling has occurred, the actual start of the model
        if dd_chilling_sum == 0:
            deacclimation = 0
        acclimation = dd_chilling_today * acclimation_rate[dormancy_period] * (
                1 - (hc_min - model_hc_yesterday) / hc_range)
        delta_hc = acclimation + deacclimation
        model_hc = model_hc_yesterday + delta_hc

        #Limit the hardiness to known min and max
        if model_hc <= hc_max:
            model_hc = hc_max
        if model_hc > hc_min:
            model_hc = hc_min

        #Sum up chilling degree days
        dd_chilling_sum = dd_chilling_sum + dd_chilling_today
        base10_chilling_sum = base10_chilling_sum + base10_chilling_today

        #Sum up heating degree days only if chilling requirement has been met i.e, dormancy period 2 has started
        if dormancy_period == 1:
            dd_heating_sum = dd_heating_sum + dd_heating_today

        #Determine if chilling requirement has been met
        # re-set dormancy period
        # order of this and other if statements is consistent with Ferguson et al, or V6.3 of our SAS code
        if base10_chilling_sum <= ecodormancy_boundary:
            dormancy_period = 1

        new_row = {'DATE': [row.DATE],
                   'MEAN_AT': [row.MEAN_AT],
                   'MIN_AT': [row.MIN_AT],
                   'MAX_AT': [row.MAX_AT],
                   'PREDICTED_LTE50': [round(model_hc, 2)],
                   'PREDICTED_BUDBREAK': [np.nan]}

        #Use hc_min to determine if vinifera or labrusca
        if hc_min == -1.2:    #Assume vinifera with budbreak at -2.2
            if model_hc_yesterday < -2.2:
                if model_hc >= -2.2:
                    #Cells(2, 12) = jdate      #and model_hc_yesterday < -2.2
                    new_row['PREDICTED_BUDBREAK'] = round(model_hc, 2)

        if hc_min == -2.5:    #Assume labrusca with budbreak at -6.4
            if model_hc_yesterday < -6.4:
                if model_hc >= -6.4:
                    #Cells(2, 12) = jdate
                    new_row['PREDICTED_BUDBREAK'] = round(model_hc, 2)

        df_results = pd.concat([df_results, pd.DataFrame.from_dict(new_row)], ignore_index=True)
        #Remember today's hardiness for tomorrow
        model_hc_yesterday = model_hc

    #This is not part of the Ferguson Model
    df_results['PREDICTED_LTE10'] = _complete_lte_model_prediction(
            cultivar, df_results['PREDICTED_LTE50'], 10)
    df_results['PREDICTED_LTE90'] = _complete_lte_model_prediction(
            cultivar, df_results['PREDICTED_LTE50'], 90)
    df_results[['PREDICTED_LTE10', 'PREDICTED_LTE90']] = df_results[
            ['PREDICTED_LTE10', 'PREDICTED_LTE90']].astype('float64').round(2)
    return df_results
