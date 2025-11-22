
'''
    COMP3105 Intro to Machine Learning
    Assignment 4
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 101260693
    Due: November 30th, 2025
'''

# assume: model.feature_extractor, model.label_head, model.domain_head
# dataloaders: in_loader (labeled), out_unlabeled_loader (unlabeled for domain)
# combine a batch with domain labels: domain=0 for in, 1 for out

opt_fy = torch.optim.Adam(list(model.feature_extractor.parameters()) +
                          list(model.label_head.parameters()), lr=1e-4)
opt_d  = torch.optim.Adam(model.domain_head.parameters(), lr=1e-4)

lambda_domain = 5.0   # try 1.0, 5.0, 10.0

for epoch in range(epochs):
    for (x_in, y_in), x_out in zip(in_loader, out_loader):
        # build domain batch
        x_domain = torch.cat([x_in, x_out], dim=0)
        domain_labels = torch.cat([torch.zeros(len(x_in)), torch.ones(len(x_out))]).long().to(device)

        # forward
        z_in = model.feature_extractor(x_in)
        preds = model.label_head(z_in)
        L_y = criterion(preds, y_in.to(device))

        z_domain = model.feature_extractor(x_domain)
        domain_preds = model.domain_head(z_domain.detach())  # update domain head first
        L_d_for_d = criterion(domain_preds, domain_labels)

        # 1) update domain head
        opt_d.zero_grad()
        L_d_for_d.backward()
        opt_d.step()

        # 2) update feature extractor + label head to MINIMIZE domain loss as well
        # recompute domain preds so gradients flow into feature extractor
        domain_preds_for_f = model.domain_head(z_domain)
        L_d_for_f = criterion(domain_preds_for_f, domain_labels)

        opt_fy.zero_grad()
        (L_y + lambda_domain * L_d_for_f).backward()
        opt_fy.step()
