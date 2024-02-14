import data as SwinUnetTrData
import model as SwinUnetTrModel


def main():
    device = "cuda:2"
    train_loader, val_loader = SwinUnetTrData.data_dataloaders()
    model = SwinUnetTrModel.define_model(device=device)
    model = SwinUnetTrModel.train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    val_inputs, val_labels, val_outputs = SwinUnetTrModel.evaluate_model(
        model=model,
        device=device,
        val_loader=val_loader
    )

    SwinUnetTrModel.plot_results(
        val_inputs=val_inputs,
        val_labels=val_labels,
        val_outputs=val_outputs
    )


if __name__ == "__main__":
    main()
