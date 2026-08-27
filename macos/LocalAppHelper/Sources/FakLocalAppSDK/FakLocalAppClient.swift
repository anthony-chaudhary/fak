import Foundation

public struct FakTaskRequest: Codable, Sendable {
    public let schema: String
    public let taskID: String
    public let task: String
    public let payload: Data
    public init(taskID: String, task: String, payload: Data) {
        self.schema = "fak.local-app-contract/1"; self.taskID = taskID; self.task = task; self.payload = payload
    }
}

public struct FakTaskEvent: Codable, Sendable {
    public let schema: String; public let sequence: UInt64; public let taskID: String; public let kind: String; public let reason: String?
}

public enum FakLocalAppError: Error { case invalidEndpoint, nonLoopbackEndpoint, invalidResponse(Int) }

/// A per-install capability-protected loopback client. It rejects LAN endpoints by construction.
public final class FakLocalAppClient: @unchecked Sendable {
    private let endpoint: URL; private let capability: String; private let session: URLSession
    public init(endpoint: URL, capability: String, session: URLSession = .shared) throws {
        guard endpoint.scheme == "http", let host = endpoint.host else { throw FakLocalAppError.invalidEndpoint }
        guard host == "127.0.0.1" || host == "::1" || host == "localhost" else { throw FakLocalAppError.nonLoopbackEndpoint }
        self.endpoint=endpoint; self.capability=capability; self.session=session
    }
    public func run(_ task: FakTaskRequest) async throws -> [FakTaskEvent] {
        var request=URLRequest(url:endpoint.appendingPathComponent("v1/tasks")); request.httpMethod="POST"
        request.setValue("Bearer \(capability)", forHTTPHeaderField:"Authorization")
        request.setValue("application/json", forHTTPHeaderField:"Content-Type"); request.httpBody=try JSONEncoder().encode(task)
        let (data,response)=try await session.data(for:request)
        guard let http=response as? HTTPURLResponse, http.statusCode == 200 else { throw FakLocalAppError.invalidResponse((response as? HTTPURLResponse)?.statusCode ?? 0) }
        return try JSONDecoder().decode([FakTaskEvent].self,from:data)
    }
    public func cancel(taskID: String) async throws {
        var request=URLRequest(url:endpoint.appendingPathComponent("v1/tasks/\(taskID)")); request.httpMethod="DELETE"
        request.setValue("Bearer \(capability)", forHTTPHeaderField:"Authorization")
        let (_,response)=try await session.data(for:request)
        guard let http=response as? HTTPURLResponse, (200...299).contains(http.statusCode) else { throw FakLocalAppError.invalidResponse((response as? HTTPURLResponse)?.statusCode ?? 0) }
    }
}
