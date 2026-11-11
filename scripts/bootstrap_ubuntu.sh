#!/usr/bin/env bash
# Run ONCE on a fresh Ubuntu 24.04 EC2 instance:  bash scripts/bootstrap_ubuntu.sh
# Installs swap, Docker Engine + Compose plugin, AWS CLI. Log out/in afterwards.
set -euo pipefail

echo "== 1/4 swap (4 GB) =="
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h | sed -n 1,3p

echo "== 2/4 base packages =="
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo apt-get install -y ca-certificates curl git unzip jq

echo "== 3/4 Docker Engine (official repo) =="
if ! command -v docker >/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
sudo usermod -aG docker "$USER"

echo "== 4/4 AWS CLI =="
command -v aws >/dev/null || sudo snap install aws-cli --classic

echo
echo "DONE. Now run:  exit   (log out)  then SSH back in, so the docker group applies."
