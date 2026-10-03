/* USB serial road-class outputs for the Arduino ESP32 core.
 * Configure CLASS_PINS and OUTPUTS_ENABLED only after mapping your hardware.
 * Each bit is a maintained class signal, not a motor direction or PWM command.
 */
#include <Arduino.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

constexpr bool OUTPUTS_ENABLED = false;
// cracks, speed_bumps, potholes, left_road_boundaries, right_road_boundaries
constexpr int CLASS_PINS[5] = {-1, -1, -1, -1, -1};
constexpr bool ACTIVE_HIGH[5] = {true, true, true, true, true};
constexpr uint32_t MAX_TTL_MS = 5000;

char sessionId[17] = "";
uint32_t lastSequence = 0;
uint32_t expiresAt = 0;
bool armed = false;
uint8_t currentMask = 0;
char inputLine[160];
size_t inputLength = 0;
bool overflowed = false;

uint16_t crc16(const char *text) {
  uint16_t crc = 0xffff;
  while (*text) {
    crc ^= static_cast<uint16_t>(static_cast<uint8_t>(*text++)) << 8;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}

void reply(const char *body) {
  char checksum[8];
  snprintf(checksum, sizeof(checksum), "*%04X\n", crc16(body));
  Serial.print(body);
  Serial.print(checksum);
}

void handleClassSignals(uint8_t mask) {
  // PARTNER INTEGRATION POINT: called for every accepted STATE and on safe stop.
  // Bit 0: cracks; bit 1: speed_bumps; bit 2: potholes;
  // bit 3: left_road_boundaries; bit 4: right_road_boundaries.
  // Signals are maintained states. Repeated packets are NOT new one-shot events.
  // Keep this function nonblocking so the communication watchdog can run.
  // mask == 0 clears all class requests. Define your machine's safe response here.
  if (!OUTPUTS_ENABLED) return;
  for (int i = 0; i < 5; ++i) {
    if (CLASS_PINS[i] < 0) continue;
    const bool active = (mask & (1 << i)) != 0;
    digitalWrite(CLASS_PINS[i], (active == ACTIVE_HIGH[i]) ? HIGH : LOW);
  }
}

void applyOutputs(uint8_t mask) {
  currentMask = mask;
  handleClassSignals(mask);
}

void safeStop() {
  applyOutputs(0);
  armed = false;
}

bool parseNumber(const char *text, uint32_t &value) {
  if (!text || !*text) return false;
  uint64_t result = 0;
  for (const char *p = text; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    result = result * 10 + (*p - '0');
    if (result > UINT32_MAX) return false;
  }
  value = static_cast<uint32_t>(result);
  return true;
}

bool validSession(const char *text) {
  if (!text || strlen(text) != 16) return false;
  for (int i = 0; i < 16; ++i)
    if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f')))
      return false;
  return true;
}

void processLine(char *line) {
  char *star = strrchr(line, '*');
  if (!star || strlen(star + 1) != 4) return;
  for (char *p = star + 1; *p; ++p)
    if (!isxdigit(static_cast<unsigned char>(*p))) return;
  const uint16_t receivedCrc = static_cast<uint16_t>(strtoul(star + 1, nullptr, 16));
  *star = '\0';
  if (crc16(line) != receivedCrc) return;

  // Preserve empty fields so malformed packets cannot shift command arguments.
  char *fields[6];
  size_t count = 1;
  fields[0] = line;
  for (char *p = line; *p; ++p) {
    if (*p == ',') {
      if (count == 6) return;
      *p = '\0';
      fields[count++] = p + 1;
    }
  }
  char response[100];
  if (count == 2 && strcmp(fields[0], "HELLO") == 0 && validSession(fields[1])) {
    safeStop();
    strcpy(sessionId, fields[1]);
    lastSequence = 0;
    snprintf(response, sizeof(response), "READY,%s", sessionId);
    reply(response);
    return;
  }
  if (count != 5 || strcmp(fields[0], "STATE") != 0 || !sessionId[0] ||
      strcmp(fields[1], sessionId) != 0) return;
  uint32_t sequence, mask, ttl;
  if (!parseNumber(fields[2], sequence) || !parseNumber(fields[3], mask) ||
      !parseNumber(fields[4], ttl) || sequence <= lastSequence ||
      mask > 31 || ttl < 100 || ttl > MAX_TTL_MS) return;
  lastSequence = sequence;
  applyOutputs(static_cast<uint8_t>(mask));
  expiresAt = millis() + ttl;
  armed = true;
  snprintf(response, sizeof(response), "ACK,%s,%lu,%lu", sessionId,
           static_cast<unsigned long>(sequence), static_cast<unsigned long>(mask));
  reply(response);
}

void setup() {
  if (OUTPUTS_ENABLED) {
    for (int i = 0; i < 5; ++i) {
      if (CLASS_PINS[i] < 0) continue;
      digitalWrite(CLASS_PINS[i], ACTIVE_HIGH[i] ? LOW : HIGH);
      pinMode(CLASS_PINS[i], OUTPUT);
    }
  }
  safeStop();
  Serial.begin(115200);
}

void loop() {
  if (armed && static_cast<int32_t>(millis() - expiresAt) >= 0) safeStop();
  // Bound each pass so serial noise cannot starve the watchdog.
  for (int budget = 0; budget < 64 && Serial.available(); ++budget) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      if (!overflowed) {
        inputLine[inputLength] = '\0';
        processLine(inputLine);
      }
      inputLength = 0;
      overflowed = false;
    } else if (c != '\r') {
      if (inputLength < sizeof(inputLine) - 1 && !overflowed)
        inputLine[inputLength++] = c;
      else
        overflowed = true;
    }
  }
}
