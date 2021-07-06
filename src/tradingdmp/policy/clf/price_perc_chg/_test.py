from tradingdmp.policy.clf.price_perc_chg.app_policy import ArangeScore

# import sys
# sys.path.append("../")
# from app_policy import ArangeScore

if __name__ == "__main__":
    data = {
        "ticker1": [0.1, 0.25, 0.25, 0.25, 0.5],
        "ticker2": [0.1, 0.2, 0.2, 0.3, 0.2],
        "tickerN": [0.1, 0.2, 0.3, 0.3, 0.1],
    }
    policy = ArangeScore()
    positions = policy.get_position(data)
    print(positions)
