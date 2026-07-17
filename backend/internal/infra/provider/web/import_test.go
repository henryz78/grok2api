package web

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

func TestParseImportedCredentialsAcceptsOneSSOTokenPerLine(t *testing.T) {
	adapter := &Adapter{}
	values, err := adapter.ParseImportedCredentials([]byte("token-one\nsso=token-two; other=drop\n\ntoken-one\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 {
		t.Fatalf("credentials = %#v", values)
	}
	if values[0].AccessToken != "token-one" || values[1].AccessToken != "token-two" {
		t.Fatalf("tokens = %q, %q", values[0].AccessToken, values[1].AccessToken)
	}
	for _, value := range values {
		if value.Provider != account.ProviderWeb || value.AuthType != account.AuthTypeSSO || value.WebTier != account.WebTierAuto {
			t.Fatalf("credential = %#v", value)
		}
	}
}

func TestParseImportedCredentialsRejectsOversizedPlainToken(t *testing.T) {
	adapter := &Adapter{}
	_, err := adapter.ParseImportedCredentials([]byte(strings.Repeat("x", maxSSOTokenBytes+1)))
	if err == nil {
		t.Fatal("expected oversized token error")
	}
}

func TestWebCredentialJSONUsesCurrentDocumentShape(t *testing.T) {
	adapter := &Adapter{}
	values, err := adapter.ParseImportedCredentials([]byte(`{"provider":"grok_web","accounts":[{"name":"primary","sso_token":"token-one","tier":"super","email":"user@example.com","user_id":"user-1","team_id":"team-1","cloudflare_cookies":"cf_clearance=abc; sso=drop"}]}`))
	if err != nil || len(values) != 1 || values[0].WebTier != account.WebTierSuper {
		t.Fatalf("credentials = %#v, err = %v", values, err)
	}
	if values[0].Email != "user@example.com" || values[0].UserID != "user-1" || values[0].TeamID != "team-1" {
		t.Fatalf("identity metadata = %#v", values[0])
	}
	if values[0].CloudflareCookies != "cf_clearance=abc; sso=drop" {
		t.Fatalf("cloudflare cookies = %q", values[0].CloudflareCookies)
	}
	data, err := adapter.MarshalCredentials(values)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"version"`) {
		t.Fatalf("export contains version metadata: %s", data)
	}
	var exported importDocument
	if err := json.Unmarshal(data, &exported); err != nil {
		t.Fatal(err)
	}
	if len(exported.Accounts) != 1 || exported.Accounts[0].Email != "user@example.com" || exported.Accounts[0].UserID != "user-1" || exported.Accounts[0].TeamID != "team-1" {
		t.Fatalf("exported identity metadata = %#v", exported.Accounts)
	}
	if _, err := adapter.ParseImportedCredentials([]byte(`{"basic":["token-one"]}`)); err == nil {
		t.Fatal("legacy tier pools were accepted")
	}
}

func TestParseImportedCredentialsExtractsIdentityFromJWTClaims(t *testing.T) {
	claims, err := json.Marshal(map[string]string{
		"email":   "jwt-user@example.com",
		"sub":     "jwt-user-1",
		"team_id": "jwt-team-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	token := "eyJhbGciOiJub25lIn0." + base64.RawURLEncoding.EncodeToString(claims) + ".signature"
	values, err := (&Adapter{}).ParseImportedCredentials([]byte(token))
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 {
		t.Fatalf("credentials = %#v", values)
	}
	if values[0].Email != "jwt-user@example.com" || values[0].UserID != "jwt-user-1" || values[0].TeamID != "jwt-team-1" {
		t.Fatalf("JWT identity metadata = %#v", values[0])
	}
}

func TestParseImportedCredentialsPrefersExplicitIdentityOverJWTClaims(t *testing.T) {
	claims := base64.RawURLEncoding.EncodeToString([]byte(`{"email":"jwt@example.com","sub":"jwt-user","team_id":"jwt-team"}`))
	token := "header." + claims + ".signature"
	data := []byte(`{"provider":"grok_web","accounts":[{"sso_token":"` + token + `","email":"explicit@example.com","user_id":"explicit-user","team_id":"explicit-team"}]}`)
	values, err := (&Adapter{}).ParseImportedCredentials(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 {
		t.Fatalf("credentials = %#v", values)
	}
	if values[0].Email != "explicit@example.com" || values[0].UserID != "explicit-user" || values[0].TeamID != "explicit-team" {
		t.Fatalf("explicit identity metadata = %#v", values[0])
	}
}
