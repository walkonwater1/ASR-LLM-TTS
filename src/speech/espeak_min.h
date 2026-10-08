#pragma once
/**
 * espeak-ng 最小化 C API 声明
 *
 * libespeak-ng.so.1 由 espeak-ng 包提供（x86_64: /usr/lib/x86_64-linux-gnu/,
 * aarch64: /usr/lib/aarch64-linux-gnu/），路径由 CMake 的 find_library 解析。
 * 不需要 -dev 包，直接声明所需函数即可。
 */

#ifdef __cplusplus
extern "C" {
#endif

/// 输出模式
typedef enum {
    AUDIO_OUTPUT_PLAYBACK   = 0,  // 直接播放
    AUDIO_OUTPUT_RETRIEVAL  = 1,  // 通过回调获取音频数据
    AUDIO_OUTPUT_SYNCHRONOUS = 2, // 同步播放
    AUDIO_OUTPUT_SYNCH_PLAYBACK = 3
} espeak_audio_output_t;

typedef enum {
    EE_OK = 0,
    EE_INTERNAL_ERROR = -1,
    EE_BUFFER_FULL = 1,
    EE_NOT_FOUND = 2
} espeak_ERROR;

typedef enum {
    POS_CHARACTER = 1,
    POS_WORD      = 2,
    POS_SENTENCE  = 3
} espeak_position_type_t;

enum {
    espeakCHARS_AUTO = 0,
    espeakCHARS_UTF8 = 1,
    espeakCHARS_8BIT = 2,
    espeakCHARS_WCHAR = 3
};

enum {
    espeakRATE       = 1,
    espeakVOLUME     = 2,
    espeakPITCH      = 3,
    espeakRANGE      = 4,
    espeakPUNCTUATION = 5,
    espeakCAPITALS   = 6
};

typedef struct {
    const char* name;
    const char* languages;
    const char* identifier;
    unsigned char gender;   // 0=unknown, 1=female, 2=male
    unsigned char age;
    unsigned char variant;
    unsigned char xx1;
    int score;
    int spare;
} espeak_VOICE;

typedef struct {
    int type;
    unsigned int text_position;
    unsigned int length;
    unsigned int audio_position;
    int sample;
    void* user_data;
} espeak_EVENT;

typedef int (*t_espeak_callback)(short* wav, int numsamples, espeak_EVENT* events);

espeak_ERROR espeak_Initialize(espeak_audio_output_t output,
                               int buflength, const char* path, int options);
espeak_ERROR espeak_SetVoiceByName(const char* name);
espeak_ERROR espeak_SetVoiceByProperties(espeak_VOICE* voice);
espeak_ERROR espeak_SetParameter(int parameter, int value, int relative);
espeak_ERROR espeak_Synth(const void* text, size_t size,
                          unsigned int position, espeak_position_type_t position_type,
                          unsigned int end_position, unsigned int flags,
                          unsigned int* unique_identifier, void* user_data);
espeak_ERROR espeak_Synchronize(void);
espeak_ERROR espeak_Terminate(void);
void espeak_SetSynthCallback(t_espeak_callback callback);

#ifdef __cplusplus
}
#endif
