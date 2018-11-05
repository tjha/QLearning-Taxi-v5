from gym.envs.registration import register

register(
    id='Taxi-v4',
    entry_point='mytaxi.mytaxi:TaxiEnv',kwargs={'jam_prob':0.1})

register(
    id='Taxi-v5',
    entry_point='mytaxi.mytaxi:TaxiEnv',kwargs={'jam_prob':0.9})
