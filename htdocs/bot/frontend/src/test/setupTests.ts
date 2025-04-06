import '@testing-library/jest-dom';
import '@testing-library/user-event';
import { server } from './mocks/server';
import 'whatwg-fetch';

// Establish API mocking before all tests.
beforeAll(() => server.listen());

// Reset any request handlers that we may add during the tests,
// so they don't affect other tests.
afterEach(() => server.resetHandlers());

// Clean up after the tests are finished.
afterAll(() => server.close());

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  },
  writable: true,
});

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver
window.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock ResizeObserver
window.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock WebSocket with proper TypeScript types
class WebSocketMock implements WebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static readonly BINARY_TYPE = 'arraybuffer' as const;

  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSING = 2;
  readonly CLOSED = 3;

  readyState = 1;
  bufferedAmount = 0;
  extensions = '';
  protocol = '';
  url = '';
  binaryType: BinaryType = 'arraybuffer';

  constructor(url: string | URL, protocols?: string | string[]) {
    this.url = typeof url === 'string' ? url : url.toString();
    if (protocols) {
      if (typeof protocols === 'string') {
        this.protocol = protocols;
      } else if (Array.isArray(protocols) && protocols.length > 0) {
        this.protocol = protocols[0];
      }
    }
  }

  send = jest.fn();
  close = jest.fn();
  onopen = jest.fn();
  onclose = jest.fn();
  onerror = jest.fn();
  onmessage = jest.fn();

  addEventListener = jest.fn();
  removeEventListener = jest.fn();
  dispatchEvent = jest.fn();
}

(window as { WebSocket: typeof WebSocket }).WebSocket = Object.assign(
  jest.fn((url: string | URL, protocols?: string | string[]) => new WebSocketMock(url, protocols)),
  WebSocketMock
) as typeof WebSocket;

// Mock navigator.geolocation
Object.defineProperty(navigator, 'geolocation', {
  value: {
    getCurrentPosition: jest.fn(),
    watchPosition: jest.fn(),
    clearWatch: jest.fn(),
  },
  writable: true,
});

// Mock crypto
Object.defineProperty(window, 'crypto', {
  value: {
    getRandomValues: jest.fn(),
    subtle: {
      digest: jest.fn(),
      encrypt: jest.fn(),
      decrypt: jest.fn(),
      sign: jest.fn(),
      verify: jest.fn(),
      generateKey: jest.fn(),
      deriveKey: jest.fn(),
      exportKey: jest.fn(),
      importKey: jest.fn(),
    },
  },
  writable: true,
});

// Polyfill for Node.js test environment
if (typeof TextEncoder === 'undefined') {
  const { TextEncoder, TextDecoder } = await import('util');
  (global as { TextEncoder: typeof TextEncoder }).TextEncoder = TextEncoder;
  (global as { TextDecoder: typeof TextDecoder }).TextDecoder = TextDecoder;
}

(global as { TextEncoder: typeof TextEncoder }).TextEncoder = TextEncoder;
(global as { TextDecoder: typeof TextDecoder }).TextDecoder = TextDecoder;
