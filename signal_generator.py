from hmm_model import HMMModel
from xgboost_model import XGBoostModel
from feature_engineering import add_select_features


def hmm_xgb_pipeline(train_df, test_df, params):

    assert train_df["date"].max() < test_df["date"].min(), "Lookahead bias!"

    feature_config = params.get("feature_config", {})

    #train_df = add_select_features(train_df, feature_config)
    #test_df = add_select_features(test_df, feature_config)

    hmm = HMMModel(n_components=params.get("n_components", 2))
    hmm.fit(train_df)

    train_df = hmm.predict(train_df)
    test_df = hmm.predict(test_df)

    mapping = hmm.derive_state_mapping(train_df)
    test_df = hmm.apply_state_mapping(test_df, mapping)

    xgb_model = XGBoostModel(params.get("xgb_params"))
    xgb_model.fit(train_df)

    test_df = xgb_model.predict(test_df)

    long_th = params.get("long_threshold", 0.65)
    short_th = params.get("short_threshold", 0.35)

    test_df["signal"] = 0

    test_df.loc[(test_df["regime"] == "bull") & (test_df["xgb_prob"] > long_th), "signal"] = 1
    test_df.loc[(test_df["regime"] == "bear") & (test_df["xgb_prob"] < short_th), "signal"] = -1

    return test_df, xgb_model.feature_importance()