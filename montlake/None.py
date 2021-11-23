

# Cell

if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    config = {}
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    hparams = HParams(**config)
    if hparams.data_sub == None:
        positions, randomindices = subset_data(hparams)
    else:
        positions = np.load(hparams.data_sub)
    np.save(hparams.outdir + '/positions' + hparams.name, positions)
    np.save(hparams.outdir + '/indices'+ hparams.name, randomindices)
    run_exp(positions, hparams)

# Cell

if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    config = {}
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    hparams = HParams(**config)
    if hparams.data_sub == None:
        positions, randomindices = subset_data(hparams)
    else:
        positions = np.load(hparams.data_sub)
    np.save(hparams.outdir + '/positions' + hparams.name, positions)
    np.save(hparams.outdir + '/indices'+ hparams.name, randomindices)
    run_exp(positions, hparams)

# Cell

if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    config = {}
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    hparams = HParams(**config)
    if hparams.data_sub == None:
        positions, randomindices = subset_data(hparams)
    else:
        positions = np.load(hparams.data_sub)
    np.save(hparams.outdir + '/positions' + hparams.name, positions)
    np.save(hparams.outdir + '/indices'+ hparams.name, randomindices)
    run_exp(positions, hparams)