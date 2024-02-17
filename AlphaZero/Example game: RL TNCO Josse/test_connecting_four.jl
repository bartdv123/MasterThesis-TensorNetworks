using AlphaZero
experiment = Examples.experiments["connect-four"]
session = Session(experiment, dir="sessions/connect-four")
resume!(session)