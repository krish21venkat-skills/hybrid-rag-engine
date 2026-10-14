# Orion Smart Meter Product Specifications

## Overview
The Orion SM-200 is a single-phase smart electricity meter designed for residential use. It reports consumption every 15 minutes over a cellular connection and supports remote firmware updates. The Orion SM-300 is the three-phase variant for commercial sites.

## Technical Specifications
| Parameter | Orion SM-200 | Orion SM-300 |
|---|---|---|
| Phase | Single | Three |
| Rated voltage | 230 V | 400 V |
| Accuracy class | 1.0 | 0.5S |
| Reporting interval | 15 minutes | 5 minutes |
| Battery backup | 5 years | 8 years |
| Operating temperature | -10 to 55 C | -20 to 60 C |
| Warranty | 2 years | 5 years |

## Connectivity
Both models use a 4G LTE module with automatic fallback to 2G. Data is encrypted with AES-256 and sent to the head-end system every reporting interval. If the network is unavailable, the meter stores up to 60 days of readings and uploads them once connectivity returns.

## Installation
Installation must be done by a certified technician. The SM-200 weighs 1.2 kg and mounts on a standard DIN rail. The meter must be installed at a height between 1.5 and 1.8 metres. After installation, the technician must run the commissioning test, which takes about 10 minutes.

## Troubleshooting
Display code E01 means no network signal. Code E07 means tamper detected on the terminal cover. Code E12 means the internal battery is below 10 percent and needs replacement.