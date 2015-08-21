using SupervisedLearning

X = rand(10, 10000000)
t = rand(10000000)
t2 = rand(5, 10000000)

msg("InMemoryLabeledDataSource: Instantiate from array (10x10000000) with targets (1x10000000)")
@spc_time ds = InMemoryLabeledDataSource(X, t)

msg("InMemoryLabeledDataSource: Access all features")
@spc_time tmp = features(ds)

msg("InMemoryLabeledDataSource: Access features minibatch n=1000")
@spc_time tmp = features(ds, 10000, 1000)

msg("InMemoryLabeledDataSource: Access features minibatch n=10000")
@spc_time tmp = features(ds, 10000, 10000)

msg("InMemoryLabeledDataSource: Access all targets")
@spc_time tmp = targets(ds)

msg("InMemoryLabeledDataSource: Access targets minibatch n=1000")
@spc_time tmp = targets(ds, 10000, 1000)

msg("InMemoryLabeledDataSource: Access targets minibatch n=10000")
@spc_time tmp = targets(ds, 10000, 10000)


msg("InMemoryLabeledDataSource: Instantiate from array (10x10000000) with targets (5x10000000)")
@spc_time ds = InMemoryLabeledDataSource(X, t2)

msg("InMemoryLabeledDataSource: Access all features")
@spc_time tmp = features(ds)

msg("InMemoryLabeledDataSource: Access minibatch n=1000")
@spc_time tmp = features(ds, 10000, 1000)

msg("InMemoryLabeledDataSource: Access minibatch n=10000")
@spc_time tmp = features(ds, 10000, 10000)

msg("InMemoryLabeledDataSource: Access all targets")
@spc_time tmp = targets(ds)

msg("InMemoryLabeledDataSource: Access targets minibatch n=1000")
@spc_time tmp = targets(ds, 10000, 1000)

msg("InMemoryLabeledDataSource: Access targets minibatch n=10000")
@spc_time tmp = targets(ds, 10000, 10000)

