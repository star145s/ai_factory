#!/usr/bin/env python3
import argparse
import bittensor as bt

def main():
    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Publish an on-chain metadata string to a Bittensor subnet"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=80,
        help="Subnet unique identifier (netuid)"
    )
    parser.add_argument(
        "--wallet-name",
        dest="wallet_name",
        type=str,
        required=True,
        help="Name of the coldkey wallet"
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        required=True,
        help="Name of the hotkey within the wallet"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="",
        help="The string to publish as metadata"
    )
    args = parser.parse_args()  # :contentReference[oaicite:0]{index=0}

    # 2. Load or create the wallet
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.hotkey)
    
    # 3. Connect to Subtensor (defaults to mainnet)
    subtensor = bt.subtensor()

    # 4. Commit the metadata string on-chain
    subtensor.commit(
        wallet=wallet,
        netuid=args.netuid,
        data="youtube id:" + args.metadata
    )  # :contentReference[oaicite:2]{index=2}

    print(
        f"✅ Published metadata to netuid={args.netuid} "
        f"using wallet='{args.wallet_name}', hotkey='{args.hotkey}'."
    )

if __name__ == "__main__":
    main()
