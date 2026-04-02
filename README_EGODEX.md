# EgoDex Commands

## Launch Teleop

```bash
./launch_egodex_tmux.sh --host bore.pub --port 58604
```

## SSH Tunnel

```bash
ssh -N -g -o GatewayPorts=yes -o ExitOnForwardFailure=yes -J zh2025@torch zh2025@gh102 -L 0.0.0.0:8087:127.0.0.1:8087 -L 0.0.0.0:8110:127.0.0.1:8110 -L 0.0.0.0:10505:127.0.0.1:10505 -L 0.0.0.0:15001:127.0.0.1:15001 -L 0.0.0.0:15102:127.0.0.1:15102 -L 0.0.0.0:8095:127.0.0.1:8095 -L 0.0.0.0:8100:127.0.0.1:8100 -L 0.0.0.0:8107:127.0.0.1:8107 -L 0.0.0.0:8109:127.0.0.1:8109
```
