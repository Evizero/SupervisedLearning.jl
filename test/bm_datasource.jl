using SupervisedLearning

X = rand(10, 10000000)
t = rand(10000000)
t2 = rand(5, 10000000)
ce = MultivalueClassEncoding(["V1","V2","V3"])
ce2 = OneOfKClassEncoding(["V1","V2","V3","V4","V5"])

msg("EncodedInMemoryLabeledDataSource: Instantiate from array (10x10000000) with targets (1x10000000)")
@spc_time ds = EncodedInMemoryLabeledDataSource(X, t, ce)

msg("EncodedInMemoryLabeledDataSource: Access all features")
@spc_time tmp = features(ds)

msg("EncodedInMemoryLabeledDataSource: Access features minibatch n=1000")
@spc_time tmp = features(ds, 10000, 1000)

msg("EncodedInMemoryLabeledDataSource: Access features minibatch n=10000")
@spc_time tmp = features(ds, 10000, 10000)

msg("EncodedInMemoryLabeledDataSource: Access all targets")
@spc_time tmp = targets(ds)

msg("EncodedInMemoryLabeledDataSource: Access targets minibatch n=1000")
@spc_time tmp = targets(ds, 10000, 1000)

msg("EncodedInMemoryLabeledDataSource: Access targets minibatch n=10000")
@spc_time tmp = targets(ds, 10000, 10000)


msg("EncodedInMemoryLabeledDataSource: Instantiate from array (10x10000000) with targets (5x10000000)")
@spc_time ds = EncodedInMemoryLabeledDataSource(X, t2, ce2)

msg("EncodedInMemoryLabeledDataSource: Access all features")
@spc_time tmp = features(ds)

msg("EncodedInMemoryLabeledDataSource: Access minibatch n=1000")
@spc_time tmp = features(ds, 10000, 1000)

msg("EncodedInMemoryLabeledDataSource: Access minibatch n=10000")
@spc_time tmp = features(ds, 10000, 10000)

msg("EncodedInMemoryLabeledDataSource: Access all targets")
@spc_time tmp = targets(ds)

msg("EncodedInMemoryLabeledDataSource: Access targets minibatch n=1000")
@spc_time tmp = targets(ds, 10000, 1000)

msg("EncodedInMemoryLabeledDataSource: Access targets minibatch n=10000")
@spc_time tmp = targets(ds, 10000, 10000)

