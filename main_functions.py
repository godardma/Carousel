

(u1, u2) = motion_optimal(x, y, theta, K, ubar)


input, output = get_dataset(K, ubar, N)


pytorch_model = train_network(input, output)


registrer_to_csv(pytorch_model)