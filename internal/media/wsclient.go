package media

import (
	"sync"

	"github.com/gorilla/websocket"
)

type WSClient struct {
	Conn *websocket.Conn
	mu   sync.Mutex
}

func NewWSClient(conn *websocket.Conn) *WSClient {
	return &WSClient{Conn: conn}
}

func (c *WSClient) WriteBinary(payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Conn.WriteMessage(websocket.BinaryMessage, payload)
}

func (c *WSClient) WriteText(payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Conn.WriteMessage(websocket.TextMessage, payload)
}

func (c *WSClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.Conn.Close()
}
