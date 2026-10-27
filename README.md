# ann-lib

> [!NOTE]
> This project is in a very early stage, and most features have not been implemented yet.

A library includes ANNs (e.g., CNNs, GNNs, RNNs).

## build script (for Linux, specifically Debian)

Before calling the script, you should make sure you have installed the following tools:

- gcc
- cmake

Build the program:

```bash
./script/build.sh
```

Build the program for debug:

```bash
./script/build.sh --debug
```

Build and run executable:

```bash
./script/build.sh --run
```

Run executable:

```bash
./script/run.sh
```

Run test:

```bash
./script/test.sh
```
