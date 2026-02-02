{ config, lib, pkgs, ... }:

{
  imports =
    [ # Include the results of the hardware scan.
      ./hardware-configuration.nix
    ];

boot.kernelPackages = pkgs.linuxPackages_5_4;
  hardware.opengl.enable = true;
  hardware.opengl.extraPackages = [ pkgs.rocm-opencl-icd ];

  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  networking.hostName = "rigolo";

  networking.networkmanager.enable = true;
  networking.firewall.allowedTCPPorts = [22];
   time.timeZone = "Europe/Paris";

  i18n.defaultLocale = "fr_FR.UTF-8";
  console = {
     font = "Lat2-Terminus16";
     #keyMap = "fr";
     useXkbConfig = true;
  };


  users.users.rig = {
     isNormalUser = true;
     extraGroups = [ "wheel" ];
     packages = with pkgs; [
       tree
     ];
   };

  environment.systemPackages = with pkgs; [
    vim
    wget
    pciutils
    nano
    git
  ];
  programs.zsh = {
    enable = true;

    ohMyZsh = {
      enable = true;
    };

    autosuggestions.enable = true;
    syntaxHighlighting.enable = true;

    shellInit = ''
      # Disable zsh's newuser startup script that prompts you to create
      # a ~/.z* file if missing
      zsh-newuser-install() { :; }
    '';
  };

  nixpkgs.config.allowUnfree = true;
    # Enable the OpenSSH daemon.
  services.openssh.enable = true;

  system.stateVersion = "23.11";
#  rocmTargets = ["gfx803" "gfx900" "gfx906"];

#  nixpkgs.overlays = [
#    (import (fetchGit {
#      url = "https://github.com/nixos-rocm/nixos-rocm.git";
#      ref = "master";
#    }))
#  ];

}
