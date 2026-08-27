import Foundation
import FakLocalAppSDK

// Packaging launches this app-scoped executable with an inherited loopback endpoint and
// per-install capability. The production helper refuses startup unless the launcher also
// supplies the signed host identity checked by fak's localapphelper admission boundary.
guard ProcessInfo.processInfo.environment["FAK_APP_CAPABILITY"]?.isEmpty == false,
      ProcessInfo.processInfo.environment["FAK_SIGNED_HOST_IDENTITY"]?.isEmpty == false else {
    FileHandle.standardError.write(Data("FakLocalAppHelper: missing authenticated app scope\n".utf8)); exit(78)
}
print("FakLocalAppHelper packaging spine; runtime endpoint is supplied by the signed app launcher")
