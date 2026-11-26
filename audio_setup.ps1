# Fixes Windows audio routing for the Voice Translator (no admin needed).
#   - Default speaker    -> Realtek speakers/headset (so game + Windows sound play normally)
#   - Default microphone -> Realtek microphone (so Discord hears you)
#   - "Listen to this device" on CABLE Output -> Realtek speakers
#     (Discord is sent into the cable; this plays it to your headset with no extra delay)
# Re-run it if a Windows/driver update ever changes your defaults:
#   powershell -ExecutionPolicy Bypass -File audio_setup.ps1

param(
    [string]$SpeakerMatch = "Realtek",
    [string]$MicMatch = "Realtek"
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct PropertyKey { public Guid fmtid; public int pid;
  public PropertyKey(Guid g, int p) { fmtid = g; pid = p; } }

[StructLayout(LayoutKind.Explicit, Size = 24)]
public struct PropVariant {
  [FieldOffset(0)] public ushort vt;
  [FieldOffset(8)] public short boolVal;
  [FieldOffset(8)] public IntPtr ptr;
}

[Guid("f8679f50-850a-41cf-9c72-430f290290c8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPolicyConfig {
  [PreserveSig] int GetMixFormat(string id, IntPtr fmt);
  [PreserveSig] int GetDeviceFormat(string id, int def, IntPtr fmt);
  [PreserveSig] int ResetDeviceFormat(string id);
  [PreserveSig] int SetDeviceFormat(string id, IntPtr a, IntPtr b);
  [PreserveSig] int GetProcessingPeriod(string id, int def, IntPtr a, IntPtr b);
  [PreserveSig] int SetProcessingPeriod(string id, IntPtr a);
  [PreserveSig] int GetShareMode(string id, IntPtr m);
  [PreserveSig] int SetShareMode(string id, IntPtr m);
  [PreserveSig] int GetPropertyValue([MarshalAs(UnmanagedType.LPWStr)] string id, [MarshalAs(UnmanagedType.Bool)] bool fx, ref PropertyKey key, out PropVariant pv);
  [PreserveSig] int SetPropertyValue([MarshalAs(UnmanagedType.LPWStr)] string id, [MarshalAs(UnmanagedType.Bool)] bool fx, ref PropertyKey key, ref PropVariant pv);
  [PreserveSig] int SetDefaultEndpoint([MarshalAs(UnmanagedType.LPWStr)] string id, int role);
  [PreserveSig] int SetEndpointVisibility(string id, int visible);
}

[ComImport, Guid("870af99c-171d-4f9e-af0d-e63df40c2bc9")] class CPolicyConfigClient {}

public static class AudioPolicy {
  static IPolicyConfig P() { return (IPolicyConfig)new CPolicyConfigClient(); }
  public static void SetDefault(string id) {
    var p = P();
    for (int role = 0; role < 3; role++) Marshal.ThrowExceptionForHR(p.SetDefaultEndpoint(id, role));
  }
  static readonly Guid ListenGuid = new Guid("24dbb0fc-9311-4b3d-9cf0-18ff155639d4");
  public static void SetListen(string captureId, string renderId, bool on) {
    var p = P();
    var kTarget = new PropertyKey(ListenGuid, 0);
    var target = new PropVariant { vt = 31, ptr = Marshal.StringToCoTaskMemUni(renderId) };
    Marshal.ThrowExceptionForHR(p.SetPropertyValue(captureId, false, ref kTarget, ref target));
    Marshal.FreeCoTaskMem(target.ptr);
    var kOn = new PropertyKey(ListenGuid, 1);
    var en = new PropVariant { vt = 11, boolVal = (short)(on ? -1 : 0) };
    Marshal.ThrowExceptionForHR(p.SetPropertyValue(captureId, false, ref kOn, ref en));
  }
}
"@

$base = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
$NAME = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"   # e.g. "Speakers"
$DESC = "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"   # e.g. "Realtek(R) Audio"

function Get-Endpoints([string]$flow) {
    $prefix = if ($flow -eq "Render") { "{0.0.0.00000000}." } else { "{0.0.1.00000000}." }
    Get-ChildItem "$base\$flow" | ForEach-Object {
        $state = (Get-ItemProperty $_.PSPath).DeviceState
        $props = Get-ItemProperty "$($_.PSPath)\Properties" -ErrorAction SilentlyContinue
        [pscustomobject]@{
            Id    = $prefix + $_.PSChildName
            Name  = "$($props.$NAME) ($($props.$DESC))"
            Active = ($state -eq 1)
        }
    } | Where-Object Active
}

$render  = Get-Endpoints "Render"
$capture = Get-Endpoints "Capture"

$spk   = $render  | Where-Object { $_.Name -match $SpeakerMatch -and $_.Name -notmatch "CABLE|VB-Audio" } | Select-Object -First 1
$mic   = $capture | Where-Object { $_.Name -match $MicMatch -and $_.Name -notmatch "CABLE|VB-Audio" } | Select-Object -First 1
$cable = $capture | Where-Object { $_.Name -match "CABLE Output" } | Select-Object -First 1

if (-not $spk)   { throw "No speaker matching '$SpeakerMatch'. Found: $($render.Name -join '; ')" }
if (-not $mic)   { throw "No microphone matching '$MicMatch'. Found: $($capture.Name -join '; ')" }
if (-not $cable) { throw "CABLE Output not found - is VB-Cable installed?" }

[AudioPolicy]::SetDefault($spk.Id);  "Default speaker    -> $($spk.Name)"
[AudioPolicy]::SetDefault($mic.Id);  "Default microphone -> $($mic.Name)"
[AudioPolicy]::SetListen($cable.Id, $spk.Id, $true)
"Listen: $($cable.Name) -> $($spk.Name)"
