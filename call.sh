#!/bin/bash

curl -X POST \
  # https://shawanna-unstoked-stiffledly.ngrok-free.dev/dialout \
  -H "Content-Type: application/json" \
  -d '{
        "to_number": "+XXXXXXXXXX",
        "from_number": "+XXXXXXXXXX"
      }'
